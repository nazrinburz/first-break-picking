"""
Trains the CNN picker. Splits by shot id (not by trace!) so traces from the
same shot never end up split across train/val/test - they're way too
correlated for that to give honest numbers.

Three-way split:
  train  : used for gradient updates
  val    : used for early stopping and best-checkpoint selection
  test   : held out entirely, reported once at the very end

The test set is carved out first (before any training decisions are made)
so the reported test number is honest. Default fractions: 15% val, 10% test.

Labels and predictions are in milliseconds from the start, so the training
loss (L1 = MAE) is directly comparable to the evaluation metric. No unit
conversion needed.
"""
import argparse
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import load_raw, build_gathers
from dataset import TraceDataset
from model import FBPickerCNN
from seed_utils import set_seed, make_generator
from report import plot_curves, write_summary, device_banner, Timer


def split_gathers(gathers, val_frac=0.15, test_frac=0.10, seed=0):
    """Shot-level three-way split.

    Test set is carved out first so it is never influenced by any training
    decision. Returns (train_gathers, val_gathers, test_gathers).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(gathers))

    n_test = max(1, int(len(gathers) * test_frac))
    n_val  = max(1, int(len(gathers) * val_frac))

    test_idx  = set(idx[:n_test].tolist())
    val_idx   = set(idx[n_test:n_test + n_val].tolist())

    train, val, test = [], [], []
    for i, g in enumerate(gathers):
        if i in test_idx:
            test.append(g)
        elif i in val_idx:
            val.append(g)
        else:
            train.append(g)
    return train, val, test


def train(hdf5_path, epochs=30, batch_size=64, lr=1e-3, device=None,
          max_shots=None, seed=0, val_frac=0.15, test_frac=0.10, patience=5):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_banner(device)
    print(f"seed={seed}  epochs={epochs}  patience={patience}  "
          f"batch_size={batch_size}  lr={lr}  "
          f"val_frac={val_frac}  test_frac={test_frac}")
    generator = make_generator(seed)

    raw = load_raw(hdf5_path, max_shots=max_shots, seed=seed)
    gathers = build_gathers(raw)
    train_g, val_g, test_g = split_gathers(gathers, val_frac=val_frac,
                                            test_frac=test_frac, seed=seed)

    train_ds = TraceDataset(train_g)
    val_ds   = TraceDataset(val_g,  max_offset=train_ds.max_offset)
    test_ds  = TraceDataset(test_g, max_offset=train_ds.max_offset)

    n_samples = train_ds.traces.shape[1]
    model = FBPickerCNN(n_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # MAE in ms — equals the evaluation metric

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          generator=generator)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)
    test_dl  = DataLoader(test_ds, batch_size=batch_size)

    history = {"train_mae_ms": [], "val_mae_ms": []}
    epoch_times = []

    best_val_mae = float("inf")
    best_weights = None
    epochs_no_improve = 0
    stopped_epoch = epochs  # will be updated if early stopping fires

    for epoch in range(epochs):
        with Timer() as t:
            model.train()
            running = 0.0
            for trace, offset, label in train_dl:
                trace, offset, label = (trace.to(device), offset.to(device),
                                        label.to(device))
                pred = model(trace, offset)
                loss = loss_fn(pred, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * len(trace)
            train_mae = running / len(train_ds)

            model.eval()
            running = 0.0
            with torch.no_grad():
                for trace, offset, label in val_dl:
                    trace, offset, label = (trace.to(device), offset.to(device),
                                            label.to(device))
                    running += loss_fn(model(trace, offset), label).item() * len(trace)
            val_mae = running / len(val_ds)

        history["train_mae_ms"].append(train_mae)
        history["val_mae_ms"].append(val_mae)
        epoch_times.append(t.elapsed)
        print(f"epoch {epoch+1:2d}/{epochs}  train MAE={train_mae:6.2f} ms  "
              f"val MAE={val_mae:6.2f} ms  ({t.elapsed:.1f}s)")

        # --- early stopping + best checkpoint ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch + 1
                print(f"early stopping at epoch {stopped_epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Restore best weights so what gets saved / returned is the best model
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"restored best weights (val MAE={best_val_mae:.2f} ms, "
              f"epoch {int(np.argmin(history['val_mae_ms'])) + 1})")

    # --- test set evaluation (run once, on best model) ---
    model.eval()
    running = 0.0
    with torch.no_grad():
        for trace, offset, label in test_dl:
            trace, offset, label = (trace.to(device), offset.to(device),
                                    label.to(device))
            running += loss_fn(model(trace, offset), label).item() * len(trace)
    test_mae = running / len(test_ds) if len(test_ds) > 0 else float("nan")

    return (model, train_g, val_g, test_g, train_ds.max_offset,
            history, epoch_times, device, best_val_mae, test_mae, stopped_epoch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_path")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="fbpicker.pt")
    ap.add_argument("--max_shots", type=int, default=None,
                    help="only load this many shots (randomly distributed across the entire asset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="fraction of shots to hold out for validation")
    ap.add_argument("--test_frac", type=float, default=0.10,
                    help="fraction of shots to hold out as untouched test set")
    ap.add_argument("--patience", type=int, default=5,
                    help="early stopping patience (epochs without val improvement)")
    ap.add_argument("--report_prefix", default=None,
                    help="basename for the saved curve plot / summary file, defaults to --out without extension")
    args = ap.parse_args()

    (model, train_g, val_g, test_g, max_offset,
     history, epoch_times, device,
     best_val_mae, test_mae, stopped_epoch) = train(
        args.hdf5_path, epochs=args.epochs, max_shots=args.max_shots,
        seed=args.seed, batch_size=args.batch_size,
        val_frac=args.val_frac, test_frac=args.test_frac,
        patience=args.patience)

    torch.save(model.state_dict(), args.out)
    print("saved best weights to", args.out)

    prefix = args.report_prefix or args.out.rsplit(".", 1)[0]
    plot_curves(history, f"training curve - {args.hdf5_path}", f"{prefix}_curve.png")

    best_epoch = int(np.argmin(history["val_mae_ms"])) + 1
    summary = [
        f"file: {args.hdf5_path}",
        f"device: {device}",
        f"epochs: {args.epochs}  patience: {args.patience}  "
        f"batch_size: {args.batch_size}  seed: {args.seed}  max_shots: {args.max_shots}",
        f"train shots: {len(train_g)}   val shots: {len(val_g)}   test shots: {len(test_g)}",
        f"stopped at epoch: {stopped_epoch}",
        f"avg epoch time: {np.mean(epoch_times):.1f}s   total training time: {sum(epoch_times):.1f}s",
        f"final train MAE: {history['train_mae_ms'][-1]:.2f} ms",
        f"final val MAE:   {history['val_mae_ms'][-1]:.2f} ms",
        f"best val MAE:    {best_val_mae:.2f} ms (epoch {best_epoch})",
        f"test MAE:        {test_mae:.2f} ms  ← held-out, not used during training",
    ]
    write_summary(summary, f"{prefix}_summary.txt")
    print(f"saved {prefix}_curve.png and {prefix}_summary.txt")
