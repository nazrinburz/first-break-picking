"""
Inference for a trained gather-level 2D first-break picker.

Features
--------
1. Loads the trained checkpoint.
2. Can automatically select a TEST shot from the checkpoint.
3. Can explicitly select a SHOTID.
4. Verifies TRAIN / VAL / TEST / UNSEEN status.
5. Refuses TRAIN shots unless --allow_seen is supplied.
6. Loads ONLY the requested shot from the HDF5 file.
   It does NOT load the entire seismic data_array.
7. Uses preprocessing configuration stored in the checkpoint.
8. Uses offset normalization learned during training.
9. Uses the same trace-windowing strategy as training.
10. Runs inference.
11. Reconstructs predictions for the complete shot.
12. Calculates metrics when labels are available.
13. Saves a prediction plot.

Examples
--------

Automatically use the first TEST shot:

    python inference_2d.py `
        --checkpoint fbpick_2d_final_model.pt `
        --asset brunswick `
        --data data/Brunswick_orig_1500ms_V2.hdf5 `
        --use_test_shot `
        --output brunswick_test.png


Use a specific TEST shot:

    python inference_2d.py `
        --checkpoint fbpick_2d_final_model.pt `
        --asset brunswick `
        --data data/Brunswick_orig_1500ms_V2.hdf5 `
        --shot_id 123456 `
        --output brunswick_123456.png


List all TEST shot IDs without loading the HDF5 file:

    python inference_2d.py `
        --checkpoint fbpick_2d_final_model.pt `
        --asset brunswick `
        --list_test_shots
"""

import argparse
import os

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_utils import GROUP, build_gathers
from resample import resample_gather
from model_2d import Gather2DCNN


# ================================================================
# Checkpoint
# ================================================================

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required = [
        "model_state_dict",
        "offset_scales",
        "split_metadata",
    ]

    missing = [
        key for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Checkpoint is missing required fields: "
            + ", ".join(missing)
            + ".\n"
            "This checkpoint was probably created by an older "
            "training script."
        )

    return checkpoint


# ================================================================
# Split information
# ================================================================

def get_asset_split_info(checkpoint, asset_name):
    split_metadata = checkpoint["split_metadata"]

    assets = split_metadata.get("assets", {})

    return assets.get(asset_name)


def get_test_shots(checkpoint, asset_name):
    asset_info = get_asset_split_info(
        checkpoint,
        asset_name,
    )

    if asset_info is None:
        return []

    return [
        int(x)
        for x in asset_info.get(
            "test_shots",
            [],
        )
    ]


def classify_shot(checkpoint, asset_name, shot_id):
    """
    Return one of:

        TRAIN
        VAL
        TEST
        UNSEEN
    """

    asset_info = get_asset_split_info(
        checkpoint,
        asset_name,
    )

    if asset_info is None:
        return "UNSEEN"

    sid = str(shot_id)

    train_ids = {
        str(x)
        for x in asset_info.get(
            "train_shots",
            [],
        )
    }

    val_ids = {
        str(x)
        for x in asset_info.get(
            "val_shots",
            [],
        )
    }

    test_ids = {
        str(x)
        for x in asset_info.get(
            "test_shots",
            [],
        )
    }

    if sid in train_ids:
        return "TRAIN"

    if sid in val_ids:
        return "VAL"

    if sid in test_ids:
        return "TEST"

    return "UNSEEN"


# ================================================================
# Efficient HDF5 shot loading
# ================================================================

def load_single_shot_raw(
    path,
    shot_id,
):
    """
    Load ONLY the traces belonging to one SHOTID.

    Important:
    ---------
    We do NOT call load_raw(), because load_raw() loads the
    complete seismic data_array.

    We only read the SHOTID column first. Once matching rows
    are known, only those rows are loaded from data_array and
    the metadata datasets.
    """

    requested_id = int(shot_id)

    print()
    print("Searching SHOTID index...")
    print("This reads only the SHOTID column, not seismic traces.")

    with h5py.File(path, "r") as f:

        if GROUP not in f:
            raise ValueError(
                f"HDF5 group '{GROUP}' was not found in:\n{path}"
            )

        g = f[GROUP]

        shot_id_ds = g["SHOTID"]

        # This is normally tiny compared with data_array.
        all_shot_ids = shot_id_ds[:].ravel()

        # Check whether SHOTID is sorted.
        is_sorted = (
            len(all_shot_ids) <= 1
            or np.all(
                all_shot_ids[:-1]
                <= all_shot_ids[1:]
            )
        )

        if is_sorted:

            left = np.searchsorted(
                all_shot_ids,
                requested_id,
                side="left",
            )

            right = np.searchsorted(
                all_shot_ids,
                requested_id,
                side="right",
            )

            if left == right:
                raise ValueError(
                    f"SHOTID {requested_id} was not found "
                    f"in {path}"
                )

            row_indices = np.arange(
                left,
                right,
                dtype=np.int64,
            )

        else:

            print(
                "SHOTID is not sorted; scanning the SHOTID "
                "index for matching rows..."
            )

            row_indices = np.flatnonzero(
                all_shot_ids == requested_id
            )

            if len(row_indices) == 0:
                raise ValueError(
                    f"SHOTID {requested_id} was not found "
                    f"in {path}"
                )

        print(
            f"Found {len(row_indices):,} traces "
            f"for SHOTID {requested_id}."
        )

        def read_rows(name, dtype=float):
            return (
                g[name][row_indices]
                .ravel()
                .astype(dtype)
            )

        raw = {
            "shot_id": read_rows(
                "SHOTID",
                dtype=np.int64,
            ),
            "source_x": read_rows(
                "SOURCE_X",
            ),
            "source_y": read_rows(
                "SOURCE_Y",
            ),
            "rec_x": read_rows(
                "REC_X",
            ),
            "rec_y": read_rows(
                "REC_Y",
            ),
            "samp_rate": read_rows(
                "SAMP_RATE",
            ),
            "coord_scale": read_rows(
                "COORD_SCALE",
            ),
            "fb_ms": read_rows(
                "SPARE1",
            ),
        }

        if "OFFSET" in g:
            raw["offset_hdr"] = read_rows(
                "OFFSET",
            )

        print(
            "Loading seismic data for this shot only..."
        )

        raw["data"] = (
            g["data_array"][row_indices]
            .astype(np.float32)
        )

    return raw


def load_single_shot_gather(
    path,
    shot_id,
):
    """
    Load one SHOTID and convert it into the same gather
    representation used during training.
    """

    raw = load_single_shot_raw(
        path,
        shot_id,
    )

    gathers = build_gathers(
        raw,
        min_traces=1,
    )

    if len(gathers) == 0:
        raise ValueError(
            f"SHOTID {shot_id} was found, but no valid "
            "gather could be constructed."
        )

    if len(gathers) > 1:
        raise ValueError(
            f"Expected one gather for SHOTID {shot_id}, "
            f"but constructed {len(gathers)}."
        )

    return gathers[0]


# ================================================================
# Windowing
# ================================================================

def make_window_starts(
    n_traces,
    max_traces,
    stride,
):
    """
    Same coverage rule as GatherDataset2D.
    """

    if max_traces <= 0:
        raise ValueError(
            "max_traces must be > 0."
        )

    if stride <= 0:
        raise ValueError(
            "trace stride must be > 0."
        )

    if stride > max_traces:
        raise ValueError(
            "trace stride cannot be greater than "
            "max_traces because that would leave gaps."
        )

    if n_traces <= max_traces:
        return [0]

    starts = list(
        range(
            0,
            n_traces - max_traces + 1,
            stride,
        )
    )

    last_start = n_traces - max_traces

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


# ================================================================
# Trace normalization
# ================================================================

def normalize_trace(
    trace,
    clip_percentile,
):
    """
    Match GatherDataset2D normalization.
    """

    trace = trace.astype(
        np.float32,
        copy=False,
    )

    scale = np.percentile(
        np.abs(trace),
        clip_percentile,
    )

    if scale < 1e-10:
        scale = 1.0

    trace = trace / scale

    trace = np.clip(
        trace,
        -3.0,
        3.0,
    )

    return trace


# ================================================================
# Prediction
# ================================================================

def predict_gather(
    model,
    gather,
    target_n_samples,
    max_traces,
    stride,
    offset_scale,
    clip_percentile,
    device,
):
    """
    Predict the complete gather.

    If windows overlap, predictions for a trace are averaged.
    """

    traces = gather["traces"]

    offsets = (
        gather["offsets"]
        / max(float(offset_scale), 1e-8)
    )

    n_traces = len(traces)

    prediction_sum = np.zeros(
        n_traces,
        dtype=np.float64,
    )

    prediction_count = np.zeros(
        n_traces,
        dtype=np.float64,
    )

    starts = make_window_starts(
        n_traces,
        max_traces,
        stride,
    )

    model.eval()

    with torch.no_grad():

        for start in starts:

            end = min(
                start + max_traces,
                n_traces,
            )

            window_traces = traces[
                start:end
            ]

            window_offsets = offsets[
                start:end
            ]

            n = len(window_traces)

            padded_traces = np.zeros(
                (
                    max_traces,
                    target_n_samples,
                ),
                dtype=np.float32,
            )

            padded_offsets = np.zeros(
                max_traces,
                dtype=np.float32,
            )

            padded_trace_mask = np.zeros(
                max_traces,
                dtype=np.float32,
            )

            # Normalize each real trace.
            for i in range(n):

                normalized = normalize_trace(
                    window_traces[i],
                    clip_percentile,
                )

                if len(normalized) != target_n_samples:
                    raise ValueError(
                        "Unexpected trace length after resampling: "
                        f"{len(normalized)} != "
                        f"{target_n_samples}"
                    )

                padded_traces[i] = normalized

            padded_offsets[:n] = (
                window_offsets
            )

            padded_trace_mask[:n] = 1.0

            # Channel 1: normalized waveform
            #
            # Channel 2: normalized offset
            #
            # Channel 3: real/padding trace mask

            offset_channel = np.repeat(
                padded_offsets[:, None],
                target_n_samples,
                axis=1,
            )

            trace_mask_channel = np.repeat(
                padded_trace_mask[:, None],
                target_n_samples,
                axis=1,
            )

            x = np.stack(
                [
                    padded_traces,
                    offset_channel,
                    trace_mask_channel,
                ],
                axis=0,
            )

            x = torch.from_numpy(
                x
            ).float().unsqueeze(0)

            x = x.to(device)

            pred_ms = model(x)[0]

            pred_ms = (
                pred_ms[:n]
                .cpu()
                .numpy()
            )

            prediction_sum[
                start:end
            ] += pred_ms

            prediction_count[
                start:end
            ] += 1.0

    if np.any(prediction_count == 0):
        raise RuntimeError(
            "Some traces received no prediction. "
            "Check max_traces and trace stride."
        )

    predictions = (
        prediction_sum
        / prediction_count
    )

    return predictions.astype(
        np.float32
    )


# ================================================================
# Metrics
# ================================================================

def calculate_metrics(
    predictions_ms,
    gather,
):
    """
    Calculate metrics only for labeled traces.
    """

    valid = gather["valid"]

    if not np.any(valid):
        return None

    target_ms = (
        gather["fb_ms"][valid]
    )

    pred_ms = (
        predictions_ms[valid]
    )

    errors = np.abs(
        pred_ms - target_ms
    )

    return {
        "n": int(len(errors)),
        "mae": float(
            np.mean(errors)
        ),
        "median": float(
            np.median(errors)
        ),
        "within_4": float(
            100.0
            * np.mean(errors <= 4)
        ),
        "within_8": float(
            100.0
            * np.mean(errors <= 8)
        ),
        "within_16": float(
            100.0
            * np.mean(errors <= 16)
        ),
    }


# ================================================================
# Plot
# ================================================================

def plot_prediction(
    gather,
    predictions_ms,
    output_path,
    title,
):
    """
    Plot seismic gather, ground truth and prediction.
    """

    traces = gather["traces"]

    n_traces, n_samples = (
        traces.shape
    )

    dt_ms = float(
        gather["ms_per_sample"]
    )

    time_ms = (
        np.arange(n_samples)
        * dt_ms
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # Normalize traces for visualization.
    max_abs = np.max(
        np.abs(traces),
        axis=1,
        keepdims=True,
    )

    normalized = (
        traces
        / (max_abs + 1e-10)
    )

    gain = 2.5

    for i in range(n_traces):

        wiggle = (
            normalized[i] * gain
            + i
        )

        ax.plot(
            wiggle,
            time_ms,
            linewidth=0.4,
        )

        ax.fill_betweenx(
            time_ms,
            i,
            wiggle,
            where=(wiggle > i),
            alpha=0.25,
        )

    # Ground truth.
    valid = gather["valid"]

    if np.any(valid):

        trace_indices = np.arange(
            n_traces
        )

        ax.plot(
            trace_indices[valid],
            gather["fb_ms"][valid],
            linewidth=2.0,
            label="Ground truth",
        )

    # Prediction.
    ax.plot(
        np.arange(n_traces),
        predictions_ms,
        linestyle="--",
        linewidth=2.0,
        label="2D CNN prediction",
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Trace number"
    )

    ax.set_ylabel(
        "Time (ms)"
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"saved plot: {output_path}"
    )


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run gather-level 2D first-break inference "
            "on a single seismic shot."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="trained *_model.pt checkpoint",
    )

    parser.add_argument(
        "--asset",
        required=True,
        help=(
            "asset name exactly as stored in the "
            "training checkpoint"
        ),
    )

    parser.add_argument(
        "--data",
        required=False,
        help="HDF5 file for the selected asset",
    )

    shot_group = parser.add_mutually_exclusive_group()

    shot_group.add_argument(
        "--shot_id",
        type=int,
        help="specific SHOTID to run",
    )

    shot_group.add_argument(
        "--use_test_shot",
        action="store_true",
        help=(
            "automatically select a TEST shot from "
            "the checkpoint"
        ),
    )

    parser.add_argument(
        "--test_index",
        type=int,
        default=0,
        help=(
            "index of TEST shot to use when "
            "--use_test_shot is supplied (default: 0)"
        ),
    )

    parser.add_argument(
        "--list_test_shots",
        action="store_true",
        help=(
            "list TEST shot IDs from the checkpoint "
            "without loading the HDF5 file"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="output PNG path",
    )

    parser.add_argument(
        "--allow_seen",
        action="store_true",
        help=(
            "allow inference on a TRAIN shot. "
            "Normally TRAIN shots are blocked."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        f"device: {device}"
    )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    # ------------------------------------------------------------
    # List TEST shots
    # ------------------------------------------------------------

    if args.list_test_shots:

        test_shots = get_test_shots(
            checkpoint,
            args.asset,
        )

        print()
        print("=" * 70)
        print(
            f"TEST SHOTS FOR ASSET: {args.asset}"
        )
        print("=" * 70)

        if not test_shots:
            print(
                "No TEST shots were found."
            )
        else:
            for i, sid in enumerate(
                test_shots
            ):
                print(
                    f"{i}: {sid}"
                )

            print()
            print(
                f"Total TEST shots: "
                f"{len(test_shots)}"
            )

        return

    # ------------------------------------------------------------
    # Determine SHOTID
    # ------------------------------------------------------------

    if args.use_test_shot:

        test_shots = get_test_shots(
            checkpoint,
            args.asset,
        )

        if not test_shots:
            raise ValueError(
                f"No TEST shots are stored in the "
                f"checkpoint for asset '{args.asset}'."
            )

        if (
            args.test_index < 0
            or args.test_index >= len(test_shots)
        ):
            raise ValueError(
                f"--test_index {args.test_index} is invalid. "
                f"Available indices: "
                f"0 to {len(test_shots) - 1}."
            )

        shot_id = test_shots[
            args.test_index
        ]

    elif args.shot_id is not None:

        shot_id = args.shot_id

    else:

        raise ValueError(
            "You must provide either "
            "--shot_id, --use_test_shot, "
            "or --list_test_shots."
        )

    # ------------------------------------------------------------
    # Data path required for actual inference
    # ------------------------------------------------------------

    if not args.data:
        raise ValueError(
            "--data is required when running "
            "actual inference."
        )

    # ------------------------------------------------------------
    # Classify SHOTID BEFORE loading seismic data
    # ------------------------------------------------------------

    split = classify_shot(
        checkpoint,
        args.asset,
        shot_id,
    )

    print()
    print("=" * 70)
    print(
        f"asset:  {args.asset}"
    )
    print(
        f"SHOTID: {shot_id}"
    )
    print(
        f"split:  {split}"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------

    if split == "TRAIN":

        print(
            "WARNING: this SHOTID was used during "
            "model training."
        )

        if not args.allow_seen:

            raise RuntimeError(
                "Inference stopped because this shot "
                "belongs to the TRAIN split.\n"
                "Use a TEST shot instead, or explicitly "
                "pass --allow_seen."
            )

    elif split == "VAL":

        print(
            "WARNING: this shot belongs to the "
            "VALIDATION split."
        )

        print(
            "It was not directly used for gradient "
            "updates, but it influenced model selection."
        )

    elif split == "TEST":

        print(
            "CONFIRMED: this SHOTID belongs to the "
            "held-out TEST split."
        )

    else:

        print(
            "This SHOTID was not present in the saved "
            "train/validation/test split."
        )

        print(
            "It is UNSEEN by this training run."
        )

    # ------------------------------------------------------------
    # Preprocessing configuration
    # ------------------------------------------------------------

    preprocessing = checkpoint.get(
        "preprocessing_config",
        {},
    )

    target_dt_ms = float(
        preprocessing.get(
            "target_dt_ms",
            checkpoint["target_dt_ms"],
        )
    )

    target_n_samples = int(
        preprocessing.get(
            "target_n_samples",
            checkpoint["target_n_samples"],
        )
    )

    max_traces = int(
        preprocessing.get(
            "max_traces",
            checkpoint["max_traces"],
        )
    )

    trace_stride = int(
        preprocessing.get(
            "trace_stride",
            checkpoint.get(
                "trace_stride",
                max_traces,
            ),
        )
    )

    clip_percentile = float(
        preprocessing.get(
            "clip_percentile",
            checkpoint.get(
                "clip_percentile",
                99.5,
            ),
        )
    )

    # ------------------------------------------------------------
    # Offset normalization
    # ------------------------------------------------------------

    offset_scales = checkpoint[
        "offset_scales"
    ]

    if args.asset not in offset_scales:

        raise ValueError(
            f"No training offset scale exists for "
            f"asset '{args.asset}'.\n"
            f"Available assets: "
            f"{list(offset_scales.keys())}"
        )

    offset_scale = float(
        offset_scales[
            args.asset
        ]
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model_config = checkpoint.get(
        "model_config",
        {},
    )

    base_ch = int(
        model_config.get(
            "base_ch",
            16,
        )
    )

    pool_k = int(
        model_config.get(
            "pool_k",
            4,
        )
    )

    model = Gather2DCNN(
        base_ch=base_ch,
        pool_k=pool_k,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ------------------------------------------------------------
    # Load ONLY this shot
    # ------------------------------------------------------------

    gather = load_single_shot_gather(
        args.data,
        shot_id,
    )

    print()
    print(
        f"gather traces: {len(gather['traces']):,}"
    )

    print(
        f"labeled traces: "
        f"{int(gather['valid'].sum()):,}"
    )

    print(
        f"original sample interval: "
        f"{gather['ms_per_sample']:.4f} ms"
    )

    # ------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------

    gather = resample_gather(
        gather,
        target_dt_ms,
        target_n_samples,
    )

    print(
        f"resampled shape: "
        f"{gather['traces'].shape}"
    )

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    predictions_ms = predict_gather(
        model=model,
        gather=gather,
        target_n_samples=target_n_samples,
        max_traces=max_traces,
        stride=trace_stride,
        offset_scale=offset_scale,
        clip_percentile=clip_percentile,
        device=device,
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        predictions_ms,
        gather,
    )

    print()
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"traces: {len(predictions_ms):,}"
    )

    if metrics is not None:

        print(
            f"labeled traces: {metrics['n']:,}"
        )

        print(
            f"MAE: "
            f"{metrics['mae']:.2f} ms"
        )

        print(
            f"median absolute error: "
            f"{metrics['median']:.2f} ms"
        )

        print(
            f"within ±4 ms: "
            f"{metrics['within_4']:.1f}%"
        )

        print(
            f"within ±8 ms: "
            f"{metrics['within_8']:.1f}%"
        )

        print(
            f"within ±16 ms: "
            f"{metrics['within_16']:.1f}%"
        )

    else:

        print(
            "No ground-truth labels available."
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    if args.output is None:

        args.output = (
            f"inference_"
            f"{args.asset}_"
            f"{shot_id}.png"
        )

    title = (
        f"{args.asset} | "
        f"SHOTID={shot_id} | "
        f"{split}"
    )

    plot_prediction(
        gather,
        predictions_ms,
        args.output,
        title,
    )

    print()
    print(
        "Inference completed successfully."
    )


if __name__ == "__main__":
    main()