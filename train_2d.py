"""
Train and evaluate a gather-level 2D CNN first-break picker.

Modes:

1. in_domain
   Shot-level train/validation/test split within each asset.

2. leave_one_out
   Train on three assets and test on the fourth.

The model receives an entire shot gather rather than an individual trace.

Each prediction is one first-break time in milliseconds.

Outputs:

    *_model.pt
    *_curve.png
    *.txt

The checkpoint stores:

    - model weights
    - model configuration
    - preprocessing configuration
    - offset normalization scales
    - best validation MAE
    - best epoch
    - seed
    - held-out asset
    - exact train SHOTIDs
    - exact validation SHOTIDs
    - exact test SHOTIDs
"""

import argparse
import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_utils import load_raw, build_gathers
from gather_dataset_2d import GatherDataset2D, make_gather_weights
from model_2d import Gather2DCNN
from seed_utils import set_seed, make_generator
from report import plot_curves, write_summary, device_banner


# ================================================================
# Timing
# ================================================================

class Timer:

    def __enter__(self):
        self.t0 = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0


# ================================================================
# Split
# ================================================================

def split_gathers_by_shot(
    gathers,
    val_frac=0.15,
    test_frac=0.10,
    seed=0,
    min_labeled=3,
):
    """
    Split complete shots into train/validation/test.

    IMPORTANT:
        The split happens BEFORE trace-window creation.

    Therefore no windows from the same SHOTID can appear in
    different splits.

    Gathers with fewer than min_labeled labeled traces are excluded
    entirely rather than silently entering the training set.
    """

    eligible = [
        g
        for g in gathers
        if g["n_labeled"] >= min_labeled
    ]

    if len(eligible) < 3:
        raise ValueError(
            f"Need at least 3 eligible shots for splitting, "
            f"but only found {len(eligible)}."
        )

    rng = np.random.default_rng(seed)

    idx = rng.permutation(
        len(eligible)
    )

    n_test = max(
        1,
        int(len(eligible) * test_frac)
    )

    n_val = max(
        1,
        int(len(eligible) * val_frac)
    )

    # Ensure at least one training shot remains.
    if n_test + n_val >= len(eligible):
        n_test = 1
        n_val = 1

    test_ids = {
        eligible[i]["shot_id"]
        for i in idx[:n_test]
    }

    val_ids = {
        eligible[i]["shot_id"]
        for i in idx[
            n_test:n_test + n_val
        ]
    }

    train_ids = {
        g["shot_id"]
        for g in eligible
        if (
            g["shot_id"] not in test_ids
            and g["shot_id"] not in val_ids
        )
    }

    train = [
        g
        for g in eligible
        if g["shot_id"] in train_ids
    ]

    val = [
        g
        for g in eligible
        if g["shot_id"] in val_ids
    ]

    test = [
        g
        for g in eligible
        if g["shot_id"] in test_ids
    ]

    # ------------------------------------------------------------
    # Hard safety check.
    # ------------------------------------------------------------

    train_check = {
        g["shot_id"]
        for g in train
    }

    val_check = {
        g["shot_id"]
        for g in val
    }

    test_check = {
        g["shot_id"]
        for g in test
    }

    assert train_check.isdisjoint(
        val_check
    ), "SHOTID leakage: train/val overlap."

    assert train_check.isdisjoint(
        test_check
    ), "SHOTID leakage: train/test overlap."

    assert val_check.isdisjoint(
        test_check
    ), "SHOTID leakage: val/test overlap."

    return train, val, test


def collect_shot_ids(gathers):
    """
    Return a sorted list of SHOTIDs.
    """

    return sorted(
        [
            g["shot_id"]
            for g in gathers
        ],
        key=lambda x: str(x)
    )


def verify_split_metadata(
    train_by,
    val_by,
    test_by,
):
    """
    Verify that no SHOTID occurs in multiple splits.

    Verification is performed independently for every asset.
    """

    assets = set()

    assets.update(train_by.keys())
    assets.update(val_by.keys())
    assets.update(test_by.keys())

    for asset in assets:

        train_ids = set(
            collect_shot_ids(
                train_by.get(asset, [])
            )
        )

        val_ids = set(
            collect_shot_ids(
                val_by.get(asset, [])
            )
        )

        test_ids = set(
            collect_shot_ids(
                test_by.get(asset, [])
            )
        )

        assert train_ids.isdisjoint(
            val_ids
        ), f"{asset}: train/val SHOTID overlap."

        assert train_ids.isdisjoint(
            test_ids
        ), f"{asset}: train/test SHOTID overlap."

        assert val_ids.isdisjoint(
            test_ids
        ), f"{asset}: val/test SHOTID overlap."


# ================================================================
# Loading
# ================================================================

def load_all(
    asset_paths,
    max_shots=None,
    seed=0,
):
    out = {}

    for name, path in asset_paths.items():

        raw = load_raw(
            path,
            max_shots=max_shots,
            seed=seed
        )

        gathers = build_gathers(raw)

        out[name] = gathers

        n_labeled = sum(
            g["n_labeled"]
            for g in gathers
        )

        n_total = sum(
            len(g["traces"])
            for g in gathers
        )

        pct = (
            100.0 * n_labeled / n_total
            if n_total
            else 0.0
        )

        print(
            f"{name}: "
            f"{len(gathers)} shots, "
            f"{n_total:,} traces, "
            f"{n_labeled:,} labeled "
            f"({pct:.1f}%)"
        )

    return out


# ================================================================
# DataLoader
# ================================================================

def make_loader(
    dataset,
    batch_size,
    balanced,
    seed,
    num_workers=0,
):
    generator = make_generator(seed)

    if balanced:

        weights = make_gather_weights(
            dataset
        )

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ================================================================
# Masked MAE
# ================================================================

def masked_mae(
    pred,
    target,
    mask,
):
    """
    MAE only over real labeled traces.

    mask:
        1 = real labeled trace
        0 = unlabeled/padded trace
    """

    error = torch.abs(
        pred - target
    )

    denom = mask.sum().clamp_min(
        1.0
    )

    return (
        error * mask
    ).sum() / denom


# ================================================================
# Training
# ================================================================

def train_one_model(
    train_ds,
    val_ds,
    epochs,
    batch_size,
    lr,
    balanced,
    patience,
    seed,
    max_traces,
    target_n_samples,
    device=None,
):
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    device_banner(device)

    print(
        f"seed={seed} "
        f"epochs={epochs} "
        f"patience={patience} "
        f"batch_size={batch_size} "
        f"lr={lr} "
        f"balanced={balanced}"
    )

    print(
        f"train gather-windows: "
        f"{len(train_ds):,}"
    )

    print(
        f"val gather-windows:   "
        f"{len(val_ds):,}"
    )

    model = Gather2DCNN(
        base_ch=16
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_dl = make_loader(
        train_ds,
        batch_size,
        balanced,
        seed,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    history = {
        "train_mae_ms": [],
        "val_mae_ms": [],
    }

    epoch_times = []

    best_val_mae = float("inf")
    best_weights = None
    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(epochs):

        # --------------------------------------------------------
        # TIMER
        #
        # IMPORTANT:
        # timer.elapsed is only valid AFTER the with-block exits.
        # --------------------------------------------------------

        with Timer() as timer:

            # ----------------------------------------------------
            # TRAIN
            # ----------------------------------------------------

            model.train()

            train_error_sum = 0.0
            train_count = 0

            for x, target, mask in train_dl:

                x = x.to(
                    device,
                    non_blocking=True
                )

                target = target.to(
                    device,
                    non_blocking=True
                )

                mask = mask.to(
                    device,
                    non_blocking=True
                )

                pred = model(x)

                loss = masked_mae(
                    pred,
                    target,
                    mask
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                optimizer.step()

                count = mask.sum().item()

                train_error_sum += (
                    loss.item() * count
                )

                train_count += count

            train_mae = (
                train_error_sum /
                max(train_count, 1)
            )

            # ----------------------------------------------------
            # VALIDATION
            # ----------------------------------------------------

            model.eval()

            val_error_sum = 0.0
            val_count = 0

            with torch.no_grad():

                for x, target, mask in val_dl:

                    x = x.to(
                        device,
                        non_blocking=True
                    )

                    target = target.to(
                        device,
                        non_blocking=True
                    )

                    mask = mask.to(
                        device,
                        non_blocking=True
                    )

                    pred = model(x)

                    error = torch.abs(
                        pred - target
                    )

                    val_error_sum += (
                        (error * mask)
                        .sum()
                        .item()
                    )

                    val_count += (
                        mask.sum().item()
                    )

            val_mae = (
                val_error_sum /
                max(val_count, 1)
            )

            history[
                "train_mae_ms"
            ].append(train_mae)

            history[
                "val_mae_ms"
            ].append(val_mae)

        # --------------------------------------------------------
        # Timer has now exited.
        # --------------------------------------------------------

        epoch_times.append(
            timer.elapsed
        )

        print(
            f"epoch {epoch + 1:2d}/{epochs} "
            f"train MAE={train_mae:7.2f} ms "
            f"val MAE={val_mae:7.2f} ms "
            f"({timer.elapsed:.1f}s)"
        )

        # --------------------------------------------------------
        # BEST CHECKPOINT IN MEMORY
        # --------------------------------------------------------

        if val_mae < best_val_mae:

            best_val_mae = val_mae

            best_epoch = epoch + 1

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    f"early stopping at epoch "
                    f"{epoch + 1} "
                    f"(no improvement for "
                    f"{patience} epochs)"
                )

                break

    # ------------------------------------------------------------
    # Restore best model.
    # ------------------------------------------------------------

    if best_weights is not None:

        model.load_state_dict(
            best_weights
        )

        print(
            f"restored best weights: "
            f"epoch={best_epoch}, "
            f"val MAE={best_val_mae:.2f} ms"
        )

    return (
        model,
        history,
        epoch_times,
        device,
        best_val_mae,
        best_epoch,
    )


# ================================================================
# Evaluation
# ================================================================

def evaluate_dataset(
    model,
    dataset,
    batch_size=16,
):
    """
    Evaluate every real labeled trace in the dataset.

    Returns absolute errors in milliseconds.
    """

    device = next(
        model.parameters()
    ).device

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    errors = []

    model.eval()

    with torch.no_grad():

        for x, target, mask in loader:

            x = x.to(device)
            target = target.to(device)
            mask = mask.to(device)

            pred = model(x)

            err = torch.abs(
                pred - target
            )

            valid_errs = err[
                mask > 0.5
            ]

            errors.extend(
                valid_errs.cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(
        errors,
        dtype=np.float32
    )


# ================================================================
# Metrics
# ================================================================

def summarize_errors(
    label,
    errors,
):
    if len(errors) == 0:
        return [
            f"{label}: no picks evaluated"
        ]

    lines = [
        f"{label}  (n={len(errors)})",
        f"  MAE:    {errors.mean():.2f} ms",
        f"  median: {np.median(errors):.2f} ms",
    ]

    for tolerance in (
        4,
        8,
        16
    ):

        percentage = (
            100.0 *
            (errors <= tolerance).mean()
        )

        lines.append(
            f"  within {tolerance:2d} ms: "
            f"{percentage:5.1f}%"
        )

    return lines


# ================================================================
# Checkpoint
# ================================================================

def save_checkpoint(
    model,
    path,
    target_dt_ms,
    target_n_samples,
    max_traces,
    trace_stride,
    offset_scales,
    best_val_mae,
    best_epoch,
    seed,
    held_out,
    split_metadata,
):
    """
    Save model plus everything required to reproduce inference
    preprocessing and determine whether a SHOTID was seen.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "model_config": {
            "base_ch": 16,
        },

        "preprocessing_config": {
            "target_dt_ms": float(
                target_dt_ms
            ),
            "target_n_samples": int(
                target_n_samples
            ),
            "max_traces": int(
                max_traces
            ),
            "trace_stride": int(
                trace_stride
            ),
        },

        # Kept as top-level fields as well for easy access/backward
        # compatibility.
        "target_dt_ms": float(
            target_dt_ms
        ),
        "target_n_samples": int(
            target_n_samples
        ),
        "max_traces": int(
            max_traces
        ),
        "trace_stride": int(
            trace_stride
        ),

        "offset_scales": {
            str(k): float(v)
            for k, v in offset_scales.items()
        },

        "best_val_mae": float(
            best_val_mae
        ),

        "best_epoch": int(
            best_epoch
        ),

        "seed": int(seed),

        "held_out": held_out,

        # --------------------------------------------------------
        # EXACT SHOT-LEVEL SPLIT
        # --------------------------------------------------------

        "split_metadata": split_metadata,
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"saved checkpoint: {path}"
    )


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assets",
        nargs="+",
        required=True,
        help=(
            "name=path pairs, e.g. "
            "brunswick=data/Brunswick.hdf5"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "in_domain",
            "leave_one_out"
        ],
        default="leave_one_out",
    )

    parser.add_argument(
        "--target_dt_ms",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--target_n_samples",
        type=int,
        default=750,
    )

    parser.add_argument(
        "--max_traces",
        type=int,
        default=64,
        help=(
            "number of traces per 2D gather window"
        ),
    )

    parser.add_argument(
        "--trace_stride",
        type=int,
        default=64,
        help=(
            "stride between gather windows; "
            "use smaller values for overlapping windows"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--no_balance",
        action="store_true",
    )

    parser.add_argument(
        "--max_shots",
        type=int,
        default=None,
        help=(
            "maximum number of randomly selected shots "
            "per asset"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--report_prefix",
        default="fbpick_2d_report",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Basic argument validation
    # ------------------------------------------------------------

    if args.trace_stride <= 0:
        raise ValueError(
            "--trace_stride must be > 0"
        )

    if args.max_traces <= 0:
        raise ValueError(
            "--max_traces must be > 0"
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be > 0"
        )

    if args.patience <= 0:
        raise ValueError(
            "--patience must be > 0"
        )

    if not (
        0.0 <= args.val_frac < 1.0
    ):
        raise ValueError(
            "--val_frac must be in [0, 1)"
        )

    if not (
        0.0 <= args.test_frac < 1.0
    ):
        raise ValueError(
            "--test_frac must be in [0, 1)"
        )

    if (
        args.val_frac +
        args.test_frac
        >= 1.0
    ):
        raise ValueError(
            "val_frac + test_frac must be < 1."
        )

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    set_seed(
        args.seed
    )

    # ------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------

    asset_paths = dict(
        a.split("=", 1)
        for a in args.assets
    )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    all_gathers = load_all(
        asset_paths,
        max_shots=args.max_shots,
        seed=args.seed,
    )

    summary_lines = [
        f"assets: {list(asset_paths.keys())}",

        (
            f"mode: {args.mode}   "
            f"epochs: {args.epochs}   "
            f"patience: {args.patience}   "
            f"seed: {args.seed}   "
            f"max_shots: {args.max_shots}   "
            f"batch_size: {args.batch_size}"
        ),

        (
            f"target_dt_ms: {args.target_dt_ms}   "
            f"target_n_samples: "
            f"{args.target_n_samples}"
        ),

        (
            f"max_traces: {args.max_traces}   "
            f"trace_stride: {args.trace_stride}"
        ),

        "",
    ]

    # ============================================================
    # IN-DOMAIN
    # ============================================================

    if args.mode == "in_domain":

        train_by = {}
        val_by = {}
        test_by = {}

        for name, gathers in all_gathers.items():

            tr, va, te = split_gathers_by_shot(
                gathers,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
            )

            train_by[name] = tr
            val_by[name] = va
            test_by[name] = te

        # --------------------------------------------------------
        # HARD SHOTID LEAKAGE CHECK
        # --------------------------------------------------------

        verify_split_metadata(
            train_by,
            val_by,
            test_by,
        )

        # --------------------------------------------------------
        # Save exact split metadata.
        # --------------------------------------------------------

        split_metadata = {
            "mode": "in_domain",
            "assets": {},
        }

        for name in asset_paths:

            split_metadata["assets"][name] = {
                "train_shots": collect_shot_ids(
                    train_by[name]
                ),
                "val_shots": collect_shot_ids(
                    val_by[name]
                ),
                "test_shots": collect_shot_ids(
                    test_by[name]
                ),
            }

        # --------------------------------------------------------
        # Training dataset calculates offset scales.
        # --------------------------------------------------------

        train_ds = GatherDataset2D(
            train_by,
            args.target_dt_ms,
            args.target_n_samples,
            max_traces=args.max_traces,
            stride=args.trace_stride,
            training=True,
        )

        # --------------------------------------------------------
        # Validation/test REUSE TRAINING scales.
        # --------------------------------------------------------

        val_ds = GatherDataset2D(
            val_by,
            args.target_dt_ms,
            args.target_n_samples,
            max_traces=args.max_traces,
            stride=args.trace_stride,
            offset_scales=train_ds.offset_scales,
        )

        test_ds = GatherDataset2D(
            test_by,
            args.target_dt_ms,
            args.target_n_samples,
            max_traces=args.max_traces,
            stride=args.trace_stride,
            offset_scales=train_ds.offset_scales,
        )

        (
            model,
            history,
            epoch_times,
            device,
            best_val_mae,
            best_epoch,
        ) = train_one_model(
            train_ds,
            val_ds,
            args.epochs,
            args.batch_size,
            args.lr,
            balanced=not args.no_balance,
            patience=args.patience,
            seed=args.seed,
            max_traces=args.max_traces,
            target_n_samples=args.target_n_samples,
        )

        checkpoint_path = (
            f"{args.report_prefix}_model.pt"
        )

        save_checkpoint(
            model,
            checkpoint_path,
            args.target_dt_ms,
            args.target_n_samples,
            args.max_traces,
            args.trace_stride,
            train_ds.offset_scales,
            best_val_mae,
            best_epoch,
            args.seed,
            held_out=None,
            split_metadata=split_metadata,
        )

        curve_path = (
            f"{args.report_prefix}"
            f"_in_domain_curve.png"
        )

        plot_curves(
            history,
            "2D CNN in-domain training curve",
            curve_path,
        )

        val_errors = evaluate_dataset(
            model,
            val_ds,
            batch_size=args.batch_size,
        )

        test_errors = evaluate_dataset(
            model,
            test_ds,
            batch_size=args.batch_size,
        )

        summary_lines += [
            (
                f"device: {device}   "
                f"best epoch: {best_epoch}   "
                f"best val MAE: "
                f"{best_val_mae:.2f} ms   "
                f"avg epoch time: "
                f"{np.mean(epoch_times):.1f}s"
            ),
            "",
            "SHOT-LEVEL SPLIT:",
        ]

        for name in asset_paths:

            summary_lines += [
                (
                    f"  {name}: "
                    f"train={len(train_by[name])} shots, "
                    f"val={len(val_by[name])} shots, "
                    f"test={len(test_by[name])} shots"
                )
            ]

        summary_lines += [
            "",
        ]

        summary_lines += summarize_errors(
            "pooled in-domain validation",
            val_errors,
        )

        summary_lines += [""]

        summary_lines += summarize_errors(
            "pooled in-domain test",
            test_errors,
        )

    # ============================================================
    # LEAVE ONE ASSET OUT
    # ============================================================

    else:

        for held_out in asset_paths:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"held out: {held_out}"
            )

            print(
                "=" * 70
            )

            train_by = {}
            val_by = {}

            for name, gathers in all_gathers.items():

                if name == held_out:
                    continue

                tr, va, _ = split_gathers_by_shot(
                    gathers,
                    val_frac=args.val_frac,
                    test_frac=0.0,
                    seed=args.seed,
                )

                train_by[name] = tr
                val_by[name] = va

            # ----------------------------------------------------
            # Held-out asset is ONLY final test data.
            # ----------------------------------------------------

            test_gathers = [
                g
                for g in all_gathers[held_out]
                if g["n_labeled"] >= 3
            ]

            # ----------------------------------------------------
            # Build explicit split metadata.
            # ----------------------------------------------------

            split_metadata = {
                "mode": "leave_one_out",
                "held_out_asset": held_out,
                "assets": {},
            }

            for name in asset_paths:

                if name == held_out:

                    split_metadata["assets"][name] = {
                        "train_shots": [],
                        "val_shots": [],
                        "test_shots": collect_shot_ids(
                            test_gathers
                        ),
                    }

                else:

                    split_metadata["assets"][name] = {
                        "train_shots": collect_shot_ids(
                            train_by[name]
                        ),
                        "val_shots": collect_shot_ids(
                            val_by[name]
                        ),
                        "test_shots": [],
                    }

            # ----------------------------------------------------
            # Convert to common dictionary form for verification.
            # ----------------------------------------------------

            test_by = {
                held_out: test_gathers
            }

            verify_split_metadata(
                train_by,
                val_by,
                test_by,
            )

            # ----------------------------------------------------
            # Training assets determine offset scales.
            # ----------------------------------------------------

            train_ds = GatherDataset2D(
                train_by,
                args.target_dt_ms,
                args.target_n_samples,
                max_traces=args.max_traces,
                stride=args.trace_stride,
                training=True,
            )

            # ----------------------------------------------------
            # Validation reuses training scales.
            # ----------------------------------------------------

            val_ds = GatherDataset2D(
                val_by,
                args.target_dt_ms,
                args.target_n_samples,
                max_traces=args.max_traces,
                stride=args.trace_stride,
                offset_scales=train_ds.offset_scales,
            )

            # ----------------------------------------------------
            # Held-out asset reuses training scales.
            # ----------------------------------------------------

            test_ds = GatherDataset2D(
                test_by,
                args.target_dt_ms,
                args.target_n_samples,
                max_traces=args.max_traces,
                stride=args.trace_stride,
                offset_scales=train_ds.offset_scales,
            )

            (
                model,
                history,
                epoch_times,
                device,
                best_val_mae,
                best_epoch,
            ) = train_one_model(
                train_ds,
                val_ds,
                args.epochs,
                args.batch_size,
                args.lr,
                balanced=not args.no_balance,
                patience=args.patience,
                seed=args.seed,
                max_traces=args.max_traces,
                target_n_samples=args.target_n_samples,
            )

            checkpoint_path = (
                f"{args.report_prefix}_loo_"
                f"{held_out}_model.pt"
            )

            save_checkpoint(
                model,
                checkpoint_path,
                args.target_dt_ms,
                args.target_n_samples,
                args.max_traces,
                args.trace_stride,
                train_ds.offset_scales,
                best_val_mae,
                best_epoch,
                args.seed,
                held_out=held_out,
                split_metadata=split_metadata,
            )

            curve_path = (
                f"{args.report_prefix}_loo_"
                f"{held_out}_curve.png"
            )

            plot_curves(
                history,
                (
                    "2D CNN leave-one-asset-out "
                    f"(held out: {held_out})"
                ),
                curve_path,
            )

            errors = evaluate_dataset(
                model,
                test_ds,
                batch_size=args.batch_size,
            )

            summary_lines += [
                f"--- held out: {held_out} ---",

                (
                    f"device: {device}   "
                    f"best epoch: {best_epoch}   "
                    f"best val MAE: "
                    f"{best_val_mae:.2f} ms   "
                    f"avg epoch time: "
                    f"{np.mean(epoch_times):.1f}s"
                ),

            ]

            summary_lines += summarize_errors(
                f"held-out asset: {held_out}",
                errors,
            )

            summary_lines += [""]

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    report_path = (
        f"{args.report_prefix}.txt"
    )

    write_summary(
        summary_lines,
        report_path,
    )

    print(
        "\n"
        f"saved report: {report_path}"
    )

    print(
        "saved model checkpoint(s) and "
        "training curve(s)"
    )