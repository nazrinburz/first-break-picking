"""
Trains on multiple assets at once and evaluates two different questions,
which are genuinely different things and worth reporting separately:

1. "in-domain" - shot-level train/val/test split within each asset, pooled
   together. Answers: how good is the picker on data from surveys it has
   seen (different shots, same asset/geology/acquisition).

2. "leave-one-asset-out" - train on N-1 assets, evaluate on the held-out
   one entirely. Answers the actual question the task poses: does this
   generalize to a new survey with different noise / geometry / near-
   surface behavior, or did it just memorize per-asset quirks.

Both matter for the presentation - a picker that's great in-domain but
falls apart leave-one-out is a real (and common) failure mode worth
showing, not hiding.

Three-way split (in_domain mode):
  train  : gradient updates
  val    : early stopping + best-checkpoint selection
  test   : held out entirely, reported once at the end

Leave-one-out mode: the held-out asset is the test set. The remaining
assets get a train/val split so early stopping can still run.

Labels and predictions are in milliseconds from the start, so the training
loss (L1 = MAE) is directly comparable to the evaluation metric.

Every run saves a training-curve plot and appends to a summary text file
(default fbpick_report.txt).
"""
import argparse
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_utils import load_raw, build_gathers
from multi_dataset import MultiAssetDataset
from model import FBPickerCNN
from evaluate import summarize
from seed_utils import set_seed, make_generator
from report import plot_curves, write_summary, device_banner, Timer


def split_gathers_by_shot(gathers, val_frac=0.15, test_frac=0.10,
                           seed=0, min_labeled=3):
    """Shot-level three-way split.

    Only shots with at least `min_labeled` labeled traces are eligible for
    val/test - no point validating against a shot that barely has any ground
    truth (this matters most for Sudbury, at ~11% labeled overall).

    Test is carved out before val so no training decision ever touches it.
    Returns (train_gathers, val_gathers, test_gathers).
    """
    eligible = [g for g in gathers if g["n_labeled"] >= min_labeled]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eligible))

    n_test = max(1, int(len(eligible) * test_frac))
    n_val  = max(1, int(len(eligible) * val_frac))

    test_set_ids = {eligible[i]["shot_id"] for i in idx[:n_test]}
    val_set_ids  = {eligible[i]["shot_id"] for i in idx[n_test:n_test + n_val]}

    train, val, test = [], [], []
    for g in gathers:
        sid = g["shot_id"]
        if sid in test_set_ids:
            test.append(g)
        elif sid in val_set_ids:
            val.append(g)
        else:
            train.append(g)
    return train, val, test


def load_all(asset_paths, max_shots=None, seed=0):
    """asset_paths: dict {name: hdf5_path}"""
    out = {}
    for name, path in asset_paths.items():
        raw = load_raw(path, max_shots=max_shots, seed=seed)
        gathers = build_gathers(raw)
        out[name] = gathers
        n_lab = sum(g["n_labeled"] for g in gathers)
        n_tot = sum(len(g["traces"]) for g in gathers)
        print(f"{name}: {len(gathers)} shots, {n_tot} traces, "
              f"{n_lab} labeled ({100*n_lab/n_tot:.1f}%)")
    return out


def make_loader(ds, batch_size, balanced, seed=0):
    generator = make_generator(seed)
    if balanced:
        sampler = WeightedRandomSampler(ds.sample_weights(), num_samples=len(ds),
                                        replacement=True, generator=generator)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)


def train_one_model(train_ds, val_ds, target_n_samples, epochs, batch_size,
                    lr, balanced, patience=5, device=None, seed=0):
    """Train with early stopping and best-checkpoint restore.

    Returns (model_with_best_weights, history, epoch_times, device,
             best_val_mae, stopped_epoch).
    val_ds may be None (leave-one-out training assets have no test set to
    compare against during training, but we still do an internal val split
    — pass val_ds=None only if genuinely no validation data is available).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_banner(device)
    print(f"seed={seed}  epochs={epochs}  patience={patience}  "
          f"batch_size={batch_size}  lr={lr}  balanced={balanced}")
    model = FBPickerCNN(target_n_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # MAE in ms

    train_dl = make_loader(train_ds, batch_size, balanced, seed=seed)
    val_dl = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    history = {"train_mae_ms": []}
    if val_dl:
        history["val_mae_ms"] = []
    epoch_times = []

    best_val_mae = float("inf")
    best_weights = None
    epochs_no_improve = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        with Timer() as t:
            model.train()
            running = 0.0
            for trace, offset, label, _ in train_dl:
                trace, offset, label = (trace.to(device), offset.to(device),
                                        label.to(device))
                pred = model(trace, offset)
                loss = loss_fn(pred, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * len(trace)
            train_mae = running / len(train_ds)

            msg = f"epoch {epoch+1:2d}/{epochs}  train MAE={train_mae:6.2f} ms"

            if val_dl:
                model.eval()
                running = 0.0
                with torch.no_grad():
                    for trace, offset, label, _ in val_dl:
                        trace, offset, label = (trace.to(device), offset.to(device),
                                                label.to(device))
                        running += loss_fn(model(trace, offset), label).item() * len(trace)
                val_mae = running / len(val_ds)
                history["val_mae_ms"].append(val_mae)
                msg += f"  val MAE={val_mae:6.2f} ms"

                # --- early stopping + best checkpoint ---
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_weights = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        stopped_epoch = epoch + 1
                        print(msg + f"  ({t.elapsed:.1f}s)")
                        print(f"early stopping at epoch {stopped_epoch} "
                              f"(no improvement for {patience} epochs)")
                        history["train_mae_ms"].append(train_mae)
                        epoch_times.append(t.elapsed)
                        break

        history["train_mae_ms"].append(train_mae)
        epoch_times.append(t.elapsed)
        print(msg + f"  ({t.elapsed:.1f}s)")

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        best_ep = int(np.argmin(history["val_mae_ms"])) + 1
        print(f"restored best weights (val MAE={best_val_mae:.2f} ms, epoch {best_ep})")

    return model, history, epoch_times, device, best_val_mae, stopped_epoch


def eval_model_multi(model, ds):
    """Evaluate model on a dataset. Returns errors in ms.

    No ms_per_sample argument needed — labels and predictions are both
    in ms already.
    """
    device = next(model.parameters()).device
    model.eval()
    errs = []
    dl = DataLoader(ds, batch_size=128)
    with torch.no_grad():
        for trace, offset, label, _ in dl:
            trace, offset, label = (trace.to(device), offset.to(device),
                                    label.to(device))
            pred = model(trace, offset)
            errs.extend(torch.abs(pred - label).cpu().tolist())
    return np.array(errs)


def errs_to_lines(label, errs, tolerances=(4, 8, 16)):
    if len(errs) == 0:
        return [f"{label}: no picks evaluated"]
    lines = [f"{label}  (n={len(errs)})",
             f"  MAE:    {errs.mean():.2f} ms",
             f"  median: {np.median(errs):.2f} ms"]
    for tol in tolerances:
        lines.append(f"  within {tol:2d} ms: {100*(errs <= tol).mean():5.1f}%")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", required=True,
                    help="name=path pairs, e.g. brunswick=data/Brunswick.hdf5")
    ap.add_argument("--mode", choices=["in_domain", "leave_one_out"], default="in_domain")
    ap.add_argument("--target_dt_ms", type=float, default=2.0)
    ap.add_argument("--target_n_samples", type=int, default=750)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5,
                    help="early stopping patience (epochs without val improvement)")
    ap.add_argument("--no_balance", action="store_true")
    ap.add_argument("--max_shots", type=int, default=None,
                    help="only load this many shots per asset (randomly distributed across the survey)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="fraction of shots used for validation (per asset)")
    ap.add_argument("--test_frac", type=float, default=0.10,
                    help="fraction of shots held out as test (per asset, in_domain mode)")
    ap.add_argument("--report_prefix", default="fbpick_report",
                    help="basename for saved curve plot(s) and the summary text file")
    args = ap.parse_args()

    set_seed(args.seed)
    asset_paths = dict(a.split("=", 1) for a in args.assets)
    all_gathers = load_all(asset_paths, max_shots=args.max_shots, seed=args.seed)

    summary_lines = [
        f"assets: {list(asset_paths.keys())}",
        f"mode: {args.mode}   epochs: {args.epochs}   patience: {args.patience}   "
        f"seed: {args.seed}   max_shots: {args.max_shots}   batch_size: {args.batch_size}",
        "",
    ]

    if args.mode == "in_domain":
        train_by, val_by, test_by = {}, {}, {}
        for name, gathers in all_gathers.items():
            tr, va, te = split_gathers_by_shot(
                gathers, val_frac=args.val_frac, test_frac=args.test_frac,
                seed=args.seed)
            train_by[name] = tr
            val_by[name]   = va
            test_by[name]  = te

        train_ds = MultiAssetDataset(train_by, args.target_dt_ms, args.target_n_samples)
        val_ds   = MultiAssetDataset(val_by,   args.target_dt_ms, args.target_n_samples,
                                     offset_scales=train_ds.offset_scales)
        test_ds  = MultiAssetDataset(test_by,  args.target_dt_ms, args.target_n_samples,
                                     offset_scales=train_ds.offset_scales)

        model, history, epoch_times, device, best_val_mae, stopped_epoch = train_one_model(
            train_ds, val_ds, args.target_n_samples, args.epochs,
            args.batch_size, 1e-3, balanced=not args.no_balance,
            patience=args.patience, seed=args.seed)

        plot_curves(history, "in-domain (pooled) training curve",
                    f"{args.report_prefix}_in_domain_curve.png")

        val_errs  = eval_model_multi(model, val_ds)
        test_errs = eval_model_multi(model, test_ds)
        summarize("pooled in-domain val (all assets mixed)", val_errs)
        summarize("pooled in-domain test (all assets mixed)", test_errs)

        summary_lines += [
            f"device: {device}   stopped epoch: {stopped_epoch}   "
            f"best val MAE: {best_val_mae:.2f} ms   avg epoch time: {np.mean(epoch_times):.1f}s",
            "",
        ]
        summary_lines += errs_to_lines("pooled in-domain val (all assets mixed)", val_errs)
        summary_lines += [""]
        summary_lines += errs_to_lines("pooled in-domain test (held out)", test_errs)

    else:  # leave_one_out
        for held_out in asset_paths:
            print(f"\n{'='*60}\nheld out: {held_out}\n{'='*60}")

            # Training assets get a train/val split so early stopping works
            train_by, val_by = {}, {}
            for name, gathers in all_gathers.items():
                if name == held_out:
                    continue
                tr, va, _ = split_gathers_by_shot(
                    gathers, val_frac=args.val_frac, test_frac=0.0,
                    seed=args.seed)
                train_by[name] = tr
                val_by[name]   = va

            # Held-out asset: only shots with labels are useful for evaluation
            test_gathers = [g for g in all_gathers[held_out]
                            if g["n_labeled"] >= 3]

            train_ds = MultiAssetDataset(train_by, args.target_dt_ms, args.target_n_samples)
            val_ds   = MultiAssetDataset(val_by,   args.target_dt_ms, args.target_n_samples,
                                         offset_scales=train_ds.offset_scales)
            test_ds  = MultiAssetDataset({held_out: test_gathers},
                                         args.target_dt_ms, args.target_n_samples,
                                         offset_scales=train_ds.offset_scales)

            model, history, epoch_times, device, best_val_mae, stopped_epoch = train_one_model(
                train_ds, val_ds, args.target_n_samples, args.epochs,
                args.batch_size, 1e-3, balanced=not args.no_balance,
                patience=args.patience, seed=args.seed)

            plot_curves(history,
                        f"leave-one-out training curve (held out: {held_out})",
                        f"{args.report_prefix}_loo_{held_out}_curve.png")

            errs = eval_model_multi(model, test_ds)
            summarize(f"held-out asset: {held_out}", errs)

            summary_lines += [
                f"--- held out: {held_out} ---",
                f"device: {device}   stopped epoch: {stopped_epoch}   "
                f"best val MAE: {best_val_mae:.2f} ms   avg epoch time: {np.mean(epoch_times):.1f}s",
            ]
            summary_lines += errs_to_lines(f"held-out asset: {held_out}", errs)
            summary_lines += [""]

    write_summary(summary_lines, f"{args.report_prefix}.txt")
    print(f"\nsaved summary to {args.report_prefix}.txt and "
          f"curve plot(s) to {args.report_prefix}*_curve.png")