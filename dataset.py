import numpy as np
import torch
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    """One item = one labeled trace + its offset + the first-break time in ms.

    Labels are taken directly from SPARE1 (fb_ms) — predicting ms rather than
    sample indices means the training loss (L1) is immediately in the same
    unit as the evaluation metric (MAE in ms), and the model output is
    physically interpretable without knowing the asset's sample rate.

    Traces get normalized per-trace by max absolute amplitude since raw
    amplitude falls off by orders of magnitude with distance from the source.
    Without normalization the network would mostly be learning
    "large trace = close receiver" rather than actually finding the onset.
    """

    def __init__(self, gathers, max_offset=None):
        traces, offsets, labels = [], [], []
        for g in gathers:
            for i in range(len(g["traces"])):
                if not g["valid"][i]:
                    continue
                traces.append(g["traces"][i])
                offsets.append(g["offsets"][i])
                labels.append(g["fb_ms"][i])   # ms directly from SPARE1

        self.traces = np.array(traces, dtype=np.float32)
        self.offsets = np.array(offsets, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
        self.max_offset = max_offset or float(self.offsets.max())

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        denom = np.max(np.abs(trace))
        if denom < 1e-10:
            denom = 1.0
        trace = trace / denom
        offset = self.offsets[idx] / self.max_offset
        return (torch.from_numpy(trace).float(),
                torch.tensor(offset, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))
