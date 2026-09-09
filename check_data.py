"""
Run this first, on every real file, before anything else. It's cheap (just
reads the small metadata arrays, not the big data_array) and will catch
path typos, missing keys, or a surprise 5th sample rate before you've
burned time on a training run that dies halfway through.

Usage:
    python check_data.py data/Brunswick_orig_1500ms_V2.hdf5
"""
import sys
import h5py
import numpy as np

GROUP = "TRACE_DATA/DEFAULT"


def check(path):
    print(f"\n{'='*60}\n{path}\n{'='*60}")
    with h5py.File(path, "r") as f:
        g = f[GROUP]
        n_traces, n_samples = g["data_array"].shape
        shot_id = g["SHOTID"][:].ravel()
        samp_rate = g["SAMP_RATE"][:].ravel()
        fb = g["SPARE1"][:].ravel()
        has_offset = "OFFSET" in g
        has_coord_scale = "COORD_SCALE" in g

        labeled = fb > 0
        _, counts = np.unique(shot_id, return_counts=True)

        print(f"traces: {n_traces:,}   samples/trace: {n_samples}")
        print(f"sample rate(s): {np.unique(samp_rate)} us  ->  "
              f"{n_samples * np.unique(samp_rate)[0] / 1000:.0f} ms trace length")
        print(f"shots: {len(counts):,}   traces/shot: {counts.min()}-{counts.max()}")
        print(f"labeled: {labeled.sum():,} / {n_traces:,} ({100*labeled.mean():.1f}%)")
        print(f"has OFFSET field: {has_offset}   has COORD_SCALE: {has_coord_scale}")

        if not has_offset:
            print("  ! no OFFSET field - will fall back to coordinate-based "
                  "distance, double check COORD_SCALE is sane for this asset")

        est_gb = n_traces * n_samples * 4 / 1e9
        print(f"estimated size of data_array in RAM if loaded whole: {est_gb:.1f} GB")
        if est_gb > 4:
            print(f"  ! that's big - consider --max_shots for a first pass "
                  f"(see README), or make sure your machine actually has the RAM")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        check(path)
