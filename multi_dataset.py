"""
Combines gathers from several assets into one training set. Three things
this handles that the single-asset TraceDataset doesn't need to:

1. Different sample rates / trace lengths -> resample everything onto one
   shared grid first (see resample.py).
2. Outlier spikes -> some traces have a handful of samples with amplitude
   orders of magnitude above the rest of the trace (dead channel noise,
   cable strikes, whatever). Dividing by the raw max, like the single-asset
   dataset does, would crush the actual signal in those traces down to
   near-zero. Using a high percentile instead of the true max is a lot more
   robust to a single bad sample.
3. Offset scale varies a lot between surveys (different survey sizes,
   different coordinate units in some cases) - normalizing offset by a
   single global max mixes those scales up. Normalizing per-asset instead
   keeps the meaningful part (how far is this receiver, relative to the
   rest of this survey) without needing the absolute distances to be
   comparable across assets.

Labels are stored in milliseconds (fb_ms from SPARE1) so the training loss
is directly in ms units — no sample-rate conversion needed, and the model
output is interpretable without knowing an asset's sample spacing.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from resample import resample_gather


class MultiAssetDataset(Dataset):
    def __init__(self, gathers_by_asset, target_dt_ms, target_n_samples,
                 clip_percentile=99.5, offset_scales=None):
        """gathers_by_asset: dict {asset_name: list_of_gathers}"""
        self.target_n_samples = target_n_samples
        self.asset_names = list(gathers_by_asset.keys())
        self.asset_to_id = {name: i for i, name in enumerate(self.asset_names)}

        # offset_scales lets you pass in scales computed on a train split
        # and reuse them for val/test, so val/test never influences
        # normalization stats
        self.offset_scales = offset_scales or {}

        traces, offsets, labels, asset_ids = [], [], [], []
        for name, gathers in gathers_by_asset.items():
            resampled = [resample_gather(g, target_dt_ms, target_n_samples) for g in gathers]

            if name not in self.offset_scales:
                all_off = np.concatenate([g["offsets"] for g in resampled])
                self.offset_scales[name] = float(np.percentile(all_off, 99)) or 1.0
            scale = self.offset_scales[name]

            for g in resampled:
                for i in range(len(g["traces"])):
                    if not g["valid"][i]:
                        continue
                    traces.append(g["traces"][i])
                    offsets.append(g["offsets"][i] / scale)
                    labels.append(g["fb_ms"][i])   # ms directly from SPARE1
                    asset_ids.append(self.asset_to_id[name])

        self.traces = np.array(traces, dtype=np.float32)
        self.offsets = np.array(offsets, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
        self.asset_ids = np.array(asset_ids, dtype=np.int64)
        self.clip_percentile = clip_percentile

    def sample_weights(self):
        """1 / (count of that trace's asset) per sample, for a
        WeightedRandomSampler - so a small asset like Sudbury doesn't get
        drowned out by a big one like Brunswick just because it has 20x
        fewer labeled traces."""
        counts = np.bincount(self.asset_ids)
        w = 1.0 / counts[self.asset_ids]
        return torch.as_tensor(w, dtype=torch.double)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        scale = np.percentile(np.abs(trace), self.clip_percentile)
        scale = scale if scale > 1e-10 else 1.0
        trace = np.clip(trace / scale, -3, 3)  # clip the residual spike tail too

        return (torch.from_numpy(trace).float(),
                torch.tensor(self.offsets[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.tensor(self.asset_ids[idx], dtype=torch.long))
