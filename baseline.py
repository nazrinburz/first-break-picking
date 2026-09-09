"""
Classic first break picker: STA/LTA ratio to get a rough pick, then an AIC
(Akaike Information Criterion) refinement to sharpen it. No training data
needed - this is basically what seismic processors used for decades before
ML pickers became common, so it's the natural thing to compare a model
against rather than a random guess.

STA/LTA idea: divide a short-term average of the signal energy by a
long-term average. Before the wave arrives you've basically got noise, so
the ratio sits near 1. The instant the wave hits, the short-term average
spikes while the long-term average hasn't caught up yet, so the ratio jumps.
Cross a threshold -> rough pick.

AIC idea: around that rough pick, find the exact sample that best splits
the local window into a "quiet" half and a "noisy" half, in a statistical
sense (minimizing that Akaike function). Tends to land a few samples closer
to the true onset than the raw STA/LTA trigger does.
"""
import numpy as np


def _trailing_avg(x, win):
    """Causal moving average: value at i only looks at samples [i-win+1, i].
    A plain np.convolve(..., mode="same") is tempting here but it's wrong -
    it's a centered window, so right at the onset the "long-term" average
    would already be peeking at post-arrival samples and the ratio never
    spikes the way it should. Near the start of the trace, where there
    aren't win samples of history yet, it just averages over what's there."""
    n = len(x)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(n)
    lo = np.maximum(0, idx - win + 1)
    return (csum[idx + 1] - csum[lo]) / (idx - lo + 1)


def sta_lta_ratio(trace, sta_len, lta_len):
    energy = trace.astype(float) ** 2
    sta = _trailing_avg(energy, sta_len)
    lta = _trailing_avg(energy, lta_len)
    lta[lta < 1e-10] = 1e-10
    return sta / lta


def pick_stalta(trace, sta_len=5, lta_len=25, thresh=3.5, persist=3):
    """persist = how many consecutive samples the ratio has to stay above
    threshold before we trust the trigger. Without this, noisy traces
    throw off single-sample spikes way before the real arrival and you
    end up "picking" random noise near the start of the trace."""
    ratio = sta_lta_ratio(trace, sta_len, lta_len)
    above = ratio > thresh
    run = 0
    for i, ok in enumerate(above):
        run = run + 1 if ok else 0
        if run >= persist:
            return i - persist + 1
    return None


def refine_aic(trace, rough_pick, half_window=40):
    n = len(trace)
    lo = max(1, rough_pick - half_window)
    hi = min(n - 1, rough_pick + half_window)
    if hi - lo < 6:
        return rough_pick

    window = trace[lo:hi].astype(float)
    aic = np.full(len(window), np.inf)
    for k in range(2, len(window) - 2):
        var1 = np.var(window[:k])
        var2 = np.var(window[k:])
        var1 = var1 if var1 > 1e-12 else 1e-12
        var2 = var2 if var2 > 1e-12 else 1e-12
        aic[k] = k * np.log(var1) + (len(window) - k - 1) * np.log(var2)

    return lo + int(np.argmin(aic))


def pick_trace(trace, sta_len=5, lta_len=25, thresh=3.5, persist=3, refine=True):
    """Single-trace convenience wrapper - handy for inspecting one trace in
    a notebook or for the visualization scripts. For evaluating lots of
    traces at once, use pick_traces_batch() instead (below) - it's the same
    algorithm but vectorized, and dramatically faster once you're past a
    few thousand traces."""
    rough = pick_stalta(trace, sta_len, lta_len, thresh, persist)
    if rough is None:
        return None
    return refine_aic(trace, rough) if refine else rough


def pick_stalta_batch(traces, sta_len=5, lta_len=25, thresh=3.5, persist=3):
    """Same idea as pick_stalta, but does every trace in a 2D (n_traces,
    n_samples) array at once with numpy broadcasting instead of a Python
    loop. Returns (rough_picks, has_trigger) - rough_picks[i] is -1 where
    has_trigger[i] is False.

    This is the same causal cumsum trick as sta_lta_ratio/_trailing_avg,
    just done across all rows of the 2D array in one shot rather than
    calling the single-trace version in a Python for loop. The persistence
    check (ratio above threshold for `persist` consecutive samples) is done
    the same way: a rolling sum of the boolean "above threshold" array,
    computed via cumsum rather than looping sample by sample.
    """
    n_traces, n_samples = traces.shape
    energy = traces.astype(float) ** 2

    def trailing_avg_batch(x, win):
        csum = np.cumsum(x, axis=1)
        csum = np.concatenate([np.zeros((n_traces, 1)), csum], axis=1)
        idx = np.arange(n_samples)
        lo = np.maximum(0, idx - win + 1)
        return (csum[:, idx + 1] - csum[:, lo]) / (idx - lo + 1)

    sta = trailing_avg_batch(energy, sta_len)
    lta = trailing_avg_batch(energy, lta_len)
    lta[lta < 1e-10] = 1e-10
    ratio = sta / lta

    above = (ratio > thresh).astype(np.int32)
    csum = np.cumsum(above, axis=1)
    csum = np.concatenate([np.zeros((n_traces, 1), dtype=np.int32), csum], axis=1)
    idx = np.arange(n_samples)
    lo = np.maximum(0, idx - persist + 1)
    run_len = csum[:, idx + 1] - csum[:, lo]
    win_len = idx - lo + 1
    trigger = (run_len >= persist) & (win_len >= persist)

    has_trigger = trigger.any(axis=1)
    first_true = np.argmax(trigger, axis=1)  # 0 if no True, but has_trigger flags that
    rough_picks = np.where(has_trigger, first_true - persist + 1, -1)
    return rough_picks, has_trigger


def refine_aic_vectorized(window):
    """Same AIC calculation as refine_aic's inner loop, but vectorized over
    every split point k at once using prefix sums instead of a Python for
    loop recomputing variance from scratch each time. O(window) instead of
    O(window^2) - this was the single biggest time sink in the original
    implementation, since it ran once per trace."""
    n = len(window)
    x = window.astype(float)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    csum2 = np.concatenate([[0.0], np.cumsum(x ** 2)])

    k = np.arange(2, n - 2)
    n1 = k
    n2 = n - k
    mean1 = csum[k] / n1
    mean2 = (csum[-1] - csum[k]) / n2
    var1 = csum2[k] / n1 - mean1 ** 2
    var2 = (csum2[-1] - csum2[k]) / n2 - mean2 ** 2
    var1 = np.maximum(var1, 1e-12)
    var2 = np.maximum(var2, 1e-12)

    aic = n1 * np.log(var1) + (n - n1 - 1) * np.log(var2)
    if len(aic) == 0:
        return None
    return int(k[np.argmin(aic)])


def pick_traces_batch(traces, sta_len=5, lta_len=25, thresh=3.5, persist=3,
                       refine=True, half_window=40):
    """The batch equivalent of calling pick_trace() in a loop, but with the
    STA/LTA + persistence stage vectorized across the whole 2D array, and
    only the (much cheaper, now O(window) not O(window^2)) AIC refinement
    still done per trace - it genuinely needs a different small window
    per trace so it can't be fully batched the same way, but the loop body
    is now light enough that this isn't the bottleneck anymore. Returns an
    array of picks, -1 where no pick was made."""
    rough_picks, has_trigger = pick_stalta_batch(traces, sta_len, lta_len, thresh, persist)
    picks = rough_picks.copy()

    if refine:
        n_samples = traces.shape[1]
        for i in np.where(has_trigger)[0]:
            rough = rough_picks[i]
            lo = max(1, rough - half_window)
            hi = min(n_samples - 1, rough + half_window)
            if hi - lo >= 6:
                refined = refine_aic_vectorized(traces[i, lo:hi])
                if refined is not None:
                    picks[i] = lo + refined
    return picks
