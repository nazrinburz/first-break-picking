"""
Small shared helper so every training run leaves behind: a plot of the
loss/MAE curves, and a text summary of the run (config, final numbers,
timing) - the stuff you actually want for a presentation appendix or to
prove GPU vs CPU timing, without having to scroll back through terminal
output to reconstruct it after the fact.
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(history, title, path):
    """history: dict of {series_name: [values per epoch]}, e.g.
    {"train_mae_ms": [...], "val_mae_ms": [...]}"""
    plt.figure(figsize=(7, 4.5))
    for name, values in history.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=name)
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def write_summary(lines, path):
    """lines: list of strings, written as plain text - easy to paste
    straight into a slide or appendix."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


def device_banner(device):
    """Prints (and returns as a string) an unambiguous confirmation of
    what's actually running the training - so there's no doubt whether the
    GPU is really being used."""
    import torch
    if str(device).startswith("cuda"):
        name = torch.cuda.get_device_name(torch.device(device))
        msg = f"using GPU: {name} (device={device})"
    else:
        msg = f"using CPU (device={device}) - no CUDA device selected"
    print(msg)
    return msg
