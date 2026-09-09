"""Wiggle-plots a shot gather with a first-break curve on top, same idea as
Figure 3 in the task doc. Handy for a sanity check and for slides."""
import numpy as np
import matplotlib.pyplot as plt


def plot_gather(gather, picks=None, title="", ax=None, gain=2.5):
    traces = gather["traces"]
    n_tr, n_samp = traces.shape

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    norm = traces / (np.max(np.abs(traces), axis=1, keepdims=True) + 1e-10)
    for i in range(n_tr):
        wig = norm[i] * gain + i
        ax.plot(wig, np.arange(n_samp), color="black", linewidth=0.4)
        ax.fill_betweenx(np.arange(n_samp), i, wig, where=(wig > i), color="black")

    if picks is not None:
        picks = np.asarray(picks)
        valid = picks >= 0
        ax.plot(np.where(valid)[0], picks[valid], color="red", linewidth=1.6)

    ax.invert_yaxis()
    ax.set_xlabel("trace #")
    ax.set_ylabel("sample")
    ax.set_title(title)
    return ax
