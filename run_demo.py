"""
Quick end-to-end demo: builds a small synthetic hdf5 (so this runs without
the real data), trains the CNN, and prints/plots baseline vs model on the
same held-out shots. Swap SYNTHETIC=False and point PATH at a real asset
once you've got one downloaded.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_synthetic_data import make_file
from train import train
from baseline import pick_trace
from evaluate import eval_baseline, eval_model, summarize
from visualize import plot_gather
from seed_utils import set_seed
from report import plot_curves, write_summary

SYNTHETIC = True
PATH = "synthetic_asset.hdf5"
SEED = 0

if __name__ == "__main__":
    set_seed(SEED)
    if SYNTHETIC:
        make_file(PATH, seed=SEED)

    print("training CNN...")
    (model, train_g, val_g, test_g, max_offset,
     history, epoch_times, device,
     best_val_mae, test_mae, stopped_epoch) = train(
        PATH, epochs=20, seed=SEED, patience=5)

    plot_curves(history, f"training curve - {PATH}", "training_curve.png")
    best_epoch = int(np.argmin(history["val_mae_ms"])) + 1
    write_summary([
        f"file: {PATH}", f"device: {device}",
        f"stopped at epoch: {stopped_epoch}",
        f"avg epoch time: {np.mean(epoch_times):.2f}s",
        f"final train MAE: {history['train_mae_ms'][-1]:.2f} ms",
        f"final val MAE:   {history['val_mae_ms'][-1]:.2f} ms",
        f"best val MAE:    {best_val_mae:.2f} ms (epoch {best_epoch})",
        f"test MAE:        {test_mae:.2f} ms  <- held-out, not used during training",
    ], "training_summary.txt")

    base_errs  = eval_baseline(val_g)
    model_errs = eval_model(model, val_g, max_offset)
    summarize("STA/LTA + AIC baseline", base_errs)
    summarize("CNN picker (val)", model_errs)
    if len(test_g) > 0:
        test_errs = eval_model(model, test_g, max_offset)
        summarize("CNN picker (test, held-out)", test_errs)

    # --- Plot one held-out gather with both sets of picks ---
    # plot_gather expects picks in sample indices (y-axis = sample).
    # Baseline picks are already sample indices.
    # Model outputs ms → divide by ms_per_sample to get sample index.
    g = val_g[0]
    ms_per_sample = g["ms_per_sample"]

    true_picks = np.where(g["valid"], g["fb_sample"], -1)
    base_picks = np.array([p if (p := pick_trace(tr)) is not None else -1
                           for tr in g["traces"]])

    model.eval()
    model_picks_ms = []
    with torch.no_grad():
        for i, tr in enumerate(g["traces"]):
            tr_n = tr.astype(np.float32)
            denom = np.max(np.abs(tr_n)) or 1.0
            tr_n = tr_n / denom
            off = g["offsets"][i] / max_offset
            t = torch.from_numpy(tr_n).unsqueeze(0).to(device)
            o = torch.tensor([off], dtype=torch.float32).to(device)
            model_picks_ms.append(model(t, o).item())
    # Convert ms → sample index for the gather plot
    model_picks = np.round(np.array(model_picks_ms) / ms_per_sample).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    plot_gather(g, picks=true_picks,  title="manual labels",        ax=axes[0])
    plot_gather(g, picks=base_picks,  title="STA/LTA + AIC baseline", ax=axes[1])
    plot_gather(g, picks=model_picks, title="CNN model",             ax=axes[2])
    plt.tight_layout()
    plt.savefig("gather_comparison.png", dpi=130)
    print("saved gather_comparison.png, training_curve.png, training_summary.txt")
