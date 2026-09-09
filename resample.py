"""
Puts traces from different assets onto the same time grid. Needed because
Lalor is recorded at 1ms/sample and the rest at 2ms/sample, and Sudbury's
traces run 2000ms long while the others run 1500ms - "sample index 400"
means a different point in time on each of them, which is meaningless to
feed into one shared model as-is.

Kept deliberately simple: linear interpolation onto a new time axis. A
proper sinc/FFT-based resample would be more "correct" signal-processing-
wise, but linear interp is sufficient here for two reasons:

1. Anti-aliasing: downsampling Lalor from 1ms to 2ms halves the Nyquist
   from 500 Hz to 250 Hz. Aliasing of energy between 250-500 Hz is possible,
   but first breaks are broadband transients (onset of wave energy), not
   narrowband signals. Aliasing artifacts from linear interp at a 2x ratio
   are negligible for onset-detection tasks.

2. Ringing: sinc/FFT resamplers introduce Gibbs-phenomenon ringing around
   sharp transients - exactly what first-break onsets are. Linear interp
   avoids this, preserving the onset shape better for our specific task.

Going from 2ms to 1ms grids or vice versa isn't a big enough rate change
for the aliasing difference between methods to matter in practice.
"""
import numpy as np


def resample_trace(trace, orig_dt_ms, target_dt_ms, target_n_samples):
    orig_n = len(trace)
    t_orig = np.arange(orig_n) * orig_dt_ms
    t_new = np.arange(target_n_samples) * target_dt_ms
    return np.interp(t_new, t_orig, trace, left=0.0, right=0.0)


def resample_gather(gather, target_dt_ms, target_n_samples):
    """Returns a new gather dict on the common grid. fb_ms stays the same
    (it's a physical time, not a sample index) - only fb_sample gets
    recomputed against the new sample spacing."""
    orig_dt = gather["ms_per_sample"]
    traces = np.stack([
        resample_trace(tr, orig_dt, target_dt_ms, target_n_samples)
        for tr in gather["traces"]
    ])

    fb_ms = gather["fb_ms"]
    valid = gather["valid"]
    fb_sample = np.where(valid, np.round(fb_ms / target_dt_ms), -1).astype(int)
    # traces whose label now falls past the (possibly shorter) new window
    # aren't usable anymore
    valid = valid & (fb_sample < target_n_samples)

    out = dict(gather)
    out["traces"] = traces
    out["fb_sample"] = fb_sample
    out["valid"] = valid
    out["ms_per_sample"] = target_dt_ms
    return out
