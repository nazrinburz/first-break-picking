"""
Small 1D CNN that reads a whole trace + its offset and predicts one number:
the first-break time in milliseconds.

Architecture rationale:
- Three conv/batchnorm blocks squeeze the waveform into spatial features.
- POOL CHANGE: AdaptiveMaxPool1d(pool_k) instead of AdaptiveAvgPool1d(1).
  Global average pooling discards where in the trace a feature fired, which
  matters for temporal location regression. MaxPool over k=4 segments keeps
  the peak response from four equal-length windows of the trace (early /
  mid-early / mid-late / late), preserving rough temporal position at
  negligible parameter cost.
- Offset is concatenated right before the dense head because source-receiver
  distance is a near-linear predictor of first-break time for a refraction
  line - it would be wasteful to make the conv layers rediscover that.
- Output is a raw linear value (ms). No sigmoid: sigmoid saturates gradients
  near 0 and near the upper bound, and artificially constraining the output
  range hurts regression. The model predicts ms directly so the loss equals
  the evaluation metric (MAE in ms) from epoch one.
"""
import torch
import torch.nn as nn


class FBPickerCNN(nn.Module):
    def __init__(self, n_samples, base_ch=16, pool_k=4):
        """
        n_samples  : number of samples in each (possibly resampled) trace.
                     No longer used to scale the output - kept for API
                     compatibility and informational purposes only.
        base_ch    : base channel count for conv blocks (doubles each block).
        pool_k     : number of temporal segments to keep after the final conv
                     block. pool_k=4 retains coarse position information
                     (early / mid-early / mid-late / late) while keeping the
                     head tiny. Increase to 8 if you want finer position
                     resolution at the cost of a slightly larger head.
        """
        super().__init__()
        self.n_samples = n_samples
        self.pool_k = pool_k

        self.conv = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=9, padding=4),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_ch, base_ch * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(),
            # pool_k segments → retain coarse temporal position information.
            # Flatten done in forward() so pool_k is configurable at init.
            nn.AdaptiveMaxPool1d(pool_k),
        )

        # Head input: (base_ch*4 * pool_k) conv features + 1 offset scalar.
        head_in = base_ch * 4 * pool_k + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # No activation: raw linear output in ms. Sigmoid was removed
            # because it (a) saturates gradients near boundaries, and
            # (b) constrains the range to [0, n_samples] which is the wrong
            # unit when predicting ms directly.
        )

    def forward(self, trace, offset):
        # trace: (B, n_samples) → unsqueeze channel dim → (B, 1, n_samples)
        x = self.conv(trace.unsqueeze(1))   # (B, base_ch*4, pool_k)
        x = x.flatten(1)                    # (B, base_ch*4 * pool_k)
        x = torch.cat([x, offset.unsqueeze(1)], dim=1)
        return self.head(x).squeeze(1)      # (B,) — predicted first-break ms
