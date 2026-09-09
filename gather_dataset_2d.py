"""
Dataset for gather-level 2D first-break picking.

One dataset item is one fixed-size chunk of a shot gather:

    [N_traces, N_samples]

If a gather has fewer than max_traces, it is padded.

If a gather has more than max_traces, it is split into trace windows
so that evaluation can cover the entire gather.

Each input has three channels:

    channel 0: normalized seismic amplitude
    channel 1: normalized offset
    channel 2: real-trace mask

The target is one first-break time in milliseconds per trace.

Only manually labeled traces contribute to the supervised loss.
Unlabeled real traces remain in the input as spatial context.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from resample import resample_gather


class GatherDataset2D(Dataset):

    def __init__(
        self,
        gathers_by_asset,
        target_dt_ms,
        target_n_samples,
        max_traces=64,
        stride=None,
        offset_scales=None,
        training=False,
        clip_percentile=99.5,
    ):
        self.target_n_samples = target_n_samples
        self.target_dt_ms = target_dt_ms
        self.max_traces = max_traces
        self.stride = stride if stride is not None else max_traces
        self.training = training
        self.clip_percentile = clip_percentile

        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        if self.max_traces <= 0:
            raise ValueError("max_traces must be > 0")

        # These scales MUST come from training data when this dataset
        # is used for validation/test/inference.
        self.offset_scales = dict(offset_scales or {})

        self.samples = []

        for asset_name, gathers in gathers_by_asset.items():

            # ----------------------------------------------------
            # Resample all gathers to the common time grid.
            # ----------------------------------------------------

            resampled = [
                resample_gather(
                    g,
                    target_dt_ms,
                    target_n_samples
                )
                for g in gathers
            ]

            # ----------------------------------------------------
            # Calculate offset scale only when one was not supplied.
            #
            # During normal training:
            #     training=True
            #     offset_scales={}
            #
            # During validation/test:
            #     offset_scales=train_ds.offset_scales
            #
            # Therefore validation/test never calculate their own
            # statistics.
            # ----------------------------------------------------

            if asset_name not in self.offset_scales:

                all_offsets = np.concatenate(
                    [
                        g["offsets"]
                        for g in resampled
                        if len(g["offsets"]) > 0
                    ]
                )

                self.offset_scales[asset_name] = (
                    float(np.percentile(all_offsets, 99))
                    if len(all_offsets) > 0
                    else 1.0
                )

            scale = max(
                float(self.offset_scales[asset_name]),
                1e-8
            )

            # ----------------------------------------------------
            # Create trace windows.
            # ----------------------------------------------------

            for g in resampled:

                valid = g["valid"].copy()

                # Completely unlabeled gathers are not useful for
                # supervised training/evaluation.
                if valid.sum() < 1:
                    continue

                traces = g["traces"]
                offsets = g["offsets"] / scale
                labels = g["fb_ms"]

                n_traces = len(traces)

                # ------------------------------------------------
                # Window starts.
                #
                # Every trace is covered, including the tail of
                # long gathers.
                # ------------------------------------------------

                if n_traces <= self.max_traces:

                    starts = [0]

                else:

                    starts = list(
                        range(
                            0,
                            n_traces - self.max_traces + 1,
                            self.stride
                        )
                    )

                    last_start = n_traces - self.max_traces

                    if starts[-1] != last_start:
                        starts.append(last_start)

                for start in starts:

                    end = min(
                        start + self.max_traces,
                        n_traces
                    )

                    self.samples.append(
                        {
                            "asset": asset_name,
                            "shot_id": g["shot_id"],
                            "traces": traces[start:end],
                            "offsets": offsets[start:end],
                            "labels": labels[start:end],
                            "valid": valid[start:end],
                            "original_start": start,
                            "original_end": end,
                        }
                    )

        if len(self.samples) == 0:
            raise ValueError(
                "No usable labeled gather samples found."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        traces = sample["traces"]
        offsets = sample["offsets"]
        labels = sample["labels"]

        # --------------------------------------------------------
        # label_valid:
        #
        # 1 = real trace with manual first-break label
        # 0 = unlabeled trace or padding
        # --------------------------------------------------------

        label_valid = sample["valid"]

        n = len(traces)

        # --------------------------------------------------------
        # Allocate padded arrays.
        # --------------------------------------------------------

        padded_traces = np.zeros(
            (
                self.max_traces,
                self.target_n_samples
            ),
            dtype=np.float32
        )

        padded_offsets = np.zeros(
            self.max_traces,
            dtype=np.float32
        )

        padded_labels = np.zeros(
            self.max_traces,
            dtype=np.float32
        )

        # Real-trace mask:
        #
        # 1 = actual seismic trace
        # 0 = padding
        #
        padded_trace_mask = np.zeros(
            self.max_traces,
            dtype=np.float32
        )

        # Label mask:
        #
        # 1 = actual trace with ground-truth label
        # 0 = unlabeled/padding
        #
        padded_label_mask = np.zeros(
            self.max_traces,
            dtype=np.float32
        )

        normalized = np.zeros_like(
            padded_traces
        )

        # --------------------------------------------------------
        # Normalize each real trace independently.
        # --------------------------------------------------------

        for i in range(n):

            trace = traces[i].astype(
                np.float32,
                copy=False
            )

            scale = np.percentile(
                np.abs(trace),
                self.clip_percentile
            )

            if scale < 1e-10:
                scale = 1.0

            trace = trace / scale
            trace = np.clip(
                trace,
                -3.0,
                3.0
            )

            normalized[i] = trace

        # --------------------------------------------------------
        # Fill real traces.
        # --------------------------------------------------------

        padded_traces[:n] = normalized
        padded_offsets[:n] = offsets
        padded_labels[:n] = labels

        # Every real trace is usable as INPUT CONTEXT.
        padded_trace_mask[:n] = 1.0

        # Only labeled traces contribute to supervised loss.
        padded_label_mask[:n] = (
            label_valid.astype(np.float32)
        )

        # --------------------------------------------------------
        # Build the three channels.
        # --------------------------------------------------------

        offset_channel = np.repeat(
            padded_offsets[:, None],
            self.target_n_samples,
            axis=1
        )

        trace_mask_channel = np.repeat(
            padded_trace_mask[:, None],
            self.target_n_samples,
            axis=1
        )

        x = np.stack(
            [
                padded_traces,
                offset_channel,
                trace_mask_channel,
            ],
            axis=0
        )

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(padded_labels).float(),
            torch.from_numpy(padded_label_mask).float(),
        )


def make_gather_weights(dataset):
    """
    Balanced sampling by asset.

    Smaller assets receive larger sampling weights so that large
    assets such as Brunswick do not dominate training.
    """

    assets = [
        sample["asset"]
        for sample in dataset.samples
    ]

    unique, counts = np.unique(
        assets,
        return_counts=True
    )

    count_map = dict(
        zip(unique, counts)
    )

    weights = np.array(
        [
            1.0 / count_map[a]
            for a in assets
        ],
        dtype=np.float64
    )

    return torch.as_tensor(
        weights,
        dtype=torch.double
    )