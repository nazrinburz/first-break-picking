import torch
import torch.nn as nn
import torch.nn.functional as F


class Gather2DCNN(nn.Module):
    """
    Gather-level 2D CNN for seismic first-break regression.

    Input:
        [B, 3, H, T]

        B = batch size
        H = number of traces (max 64)
        T = 750 time samples

    Channels:
        0 = normalized seismic waveform
        1 = normalized source-receiver offset
        2 = real-trace mask

    Output:
        [B, H]

    One output = predicted first-break time in milliseconds
    for one trace.
    """

    def __init__(self, base_ch=16, pool_k=4):
        super().__init__()

        self.pool_k = pool_k

        self.encoder = nn.Sequential(
            # Look across neighboring traces AND nearby time samples
            nn.Conv2d(
                3, base_ch,
                kernel_size=(5, 9),
                padding=(2, 4)
            ),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),

            # Downsample TIME only.
            # Keep trace dimension unchanged.
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(
                base_ch, base_ch * 2,
                kernel_size=(5, 7),
                padding=(2, 3)
            ),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(
                base_ch * 2, base_ch * 4,
                kernel_size=(3, 5),
                padding=(1, 2)
            ),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(),

            nn.Conv2d(
                base_ch * 4, base_ch * 4,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(),
        )

        # Regression head.
        #
        # We retain 4 coarse time regions instead of
        # completely destroying temporal position.
        self.head = nn.Sequential(
            nn.Linear(base_ch * 4 * pool_k, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, 3, H, T]
        x = self.encoder(x)

        # Preserve trace dimension H.
        # Compress time to 4 coarse bins.
        x = F.adaptive_avg_pool2d(
            x,
            (x.shape[2], self.pool_k)
        )

        # [B, C, H, 4]
        x = x.permute(0, 2, 1, 3)

        # [B, H, C*4]
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1
        )

        # [B, H]
        return self.head(x).squeeze(-1)