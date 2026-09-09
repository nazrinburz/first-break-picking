"""
Runs both pickers on the same held-out shots and prints the numbers you'd
actually put in a slide: mean absolute error in ms, median error, and
"percent of picks within X ms" - that last one is the standard QC metric
people use for first break picking since a handful of badly missed picks
matters less than the bulk of picks being close.

All errors are in milliseconds. The model now predicts ms directly, so
eval_model no longer needs to multiply by ms_per_sample.
"""
import numpy as np
import torch

from baseline import pick_traces_batch


def eval_baseline(gathers):
    """Runs the batch (vectorized) picker once per gather instead of once
    per trace - see baseline.py's pick_traces_batch. This is the difference
    between minutes and seconds on a real asset."""
    errs = []
    for g in gathers:
        picks = pick_traces_batch(g["traces"])
        valid = g["valid"] & (picks >= 0)
        # Baseline picks in sample indices → convert to ms for comparison
        diff = np.abs(picks[valid] - g["fb_sample"][valid]) * g["ms_per_sample"]
        errs.append(diff)
    return np.concatenate(errs) if errs else np.array([])


def eval_model(model, gathers, max_offset):
    """Evaluate the CNN model on a list of gathers.

    The model predicts first-break time in ms directly, so errors are
    computed as |pred_ms - fb_ms| with no unit conversion.
    """
    device = next(model.parameters()).device
    model.eval()
    errs = []
    with torch.no_grad():
        for g in gathers:
            for i in range(len(g["traces"])):
                if not g["valid"][i]:
                    continue
                trace = g["traces"][i].astype(np.float32)
                denom = np.max(np.abs(trace))
                denom = denom if denom > 1e-10 else 1.0
                trace = trace / denom
                offset = g["offsets"][i] / max_offset

                t = torch.from_numpy(trace).unsqueeze(0).to(device)
                o = torch.tensor([offset], dtype=torch.float32).to(device)
                pred_ms = model(t, o).item()          # already in ms
                errs.append(abs(pred_ms - g["fb_ms"][i]))
    return np.array(errs)


def summarize(name, errs_ms, tolerances=(4, 8, 16)):
    print(f"\n{name}  (n={len(errs_ms)})")
    print(f"  MAE:    {errs_ms.mean():.2f} ms")
    print(f"  median: {np.median(errs_ms):.2f} ms")
    for tol in tolerances:
        pct = 100 * (errs_ms <= tol).mean()
        print(f"  within {tol:2d} ms: {pct:5.1f}%")
