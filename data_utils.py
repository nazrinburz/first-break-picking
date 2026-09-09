"""
Utilities to load the raw HDF5 seismic files and reorganize the flat trace
array into per-shot gathers (what the task calls "2D seismic images").

The files store every trace from every shot as one big 2D array with no
explicit grouping, so the first real step is splitting that back out into
shots and sorting each shot by offset (source-to-receiver distance) - that's
what gives you the nice fan shape from Figure 1 instead of a scrambled mess.
"""
import numpy as np
import h5py

GROUP = "TRACE_DATA/DEFAULT"


def load_raw(path, max_shots=None, seed=0):
    """Load seismic traces and metadata from HDF5.

    If max_shots is given, randomly selects that many shots across the
    survey. For sorted SHOTID data, contiguous shot ranges are read directly.

    Memory-efficient implementation:
    Instead of creating a list of large trace chunks and then calling
    np.concatenate(), we preallocate the final float32 array and fill it
    chunk by chunk. This avoids holding two copies of the large data array
    in memory at once.
    """
    with h5py.File(path, "r") as f:
        g = f[GROUP]

        shot_id = g["SHOTID"][:].ravel()
        n = len(shot_id)

        lo_indices = None
        hi_indices = None
        row_idx = None
        is_sorted = True

        if max_shots is not None:
            unique_shots = np.unique(shot_id)
            is_sorted = np.all(np.diff(shot_id) >= 0)

            k = min(max_shots, len(unique_shots))

            rng = np.random.default_rng(seed)
            chosen = rng.choice(unique_shots, size=k, replace=False)

            if is_sorted:
                chosen.sort()

                lo_indices = np.searchsorted(
                    shot_id, chosen, side="left"
                )
                hi_indices = np.searchsorted(
                    shot_id, chosen, side="right"
                )

                n_traces = int(
                    np.sum(hi_indices - lo_indices)
                )

                print(
                    f"loading {k} random shots "
                    f"({n_traces:,} traces) out of {n:,} total"
                )

            else:
                print(
                    "warning: SHOTID isn't sorted/contiguous in this file - "
                    "using scattered read, which may be slower"
                )

                row_idx = np.where(
                    np.isin(shot_id, chosen)
                )[0]

                print(
                    f"loading {k} random shots "
                    f"({len(row_idx):,} traces) out of {n:,} total"
                )

        # Helper for 1D metadata
        def read_rows(name, dtype=float):
            arr = g[name][:].ravel().astype(dtype)

            if max_shots is None:
                return arr

            if is_sorted:
                n_out = int(
                    np.sum(hi_indices - lo_indices)
                )

                result = np.empty(n_out, dtype=dtype)

                pos = 0
                for lo, hi in zip(lo_indices, hi_indices):
                    size = hi - lo
                    result[pos:pos + size] = arr[lo:hi]
                    pos += size

                return result

            return arr[row_idx]

        # Shot IDs
        if max_shots is None:
            out_shot_id = shot_id

        elif is_sorted:
            n_out = int(
                np.sum(hi_indices - lo_indices)
            )

            out_shot_id = np.empty(n_out, dtype=shot_id.dtype)

            pos = 0
            for lo, hi in zip(lo_indices, hi_indices):
                size = hi - lo
                out_shot_id[pos:pos + size] = shot_id[lo:hi]
                pos += size

        else:
            out_shot_id = shot_id[row_idx]

        # Metadata
        out = {
            "shot_id": out_shot_id,
            "source_x": read_rows("SOURCE_X"),
            "source_y": read_rows("SOURCE_Y"),
            "rec_x": read_rows("REC_X"),
            "rec_y": read_rows("REC_Y"),
            "samp_rate": read_rows("SAMP_RATE"),
            "coord_scale": read_rows("COORD_SCALE"),
            "fb_ms": read_rows("SPARE1"),
        }

        if "OFFSET" in g:
            out["offset_hdr"] = read_rows("OFFSET")

        # Large seismic data
        data_ds = g["data_array"]

        if max_shots is None:
            # Full dataset
            out["data"] = data_ds[:].astype(np.float32)

        elif is_sorted:
            # IMPORTANT:
            # Preallocate final array instead of using:
            # chunks = [...]
            # np.concatenate(chunks)
            #
            # This avoids a second huge allocation.
            n_samples = data_ds.shape[1]

            data = np.empty(
                (n_traces, n_samples),
                dtype=np.float32
            )

            pos = 0

            for lo, hi in zip(lo_indices, hi_indices):
                size = hi - lo

                data[pos:pos + size] = data_ds[lo:hi]

                pos += size

            out["data"] = data

        else:
            # Unsorted fallback
            out["data"] = data_ds[row_idx].astype(np.float32)

    return out

def apply_coord_scale(raw):
    """COORD_SCALE in SEG-Y-style headers is a signed multiplier: positive
    means multiply, negative means divide by abs(value). Some files just
    store 1 everywhere, in which case this does nothing."""
    scale = raw["coord_scale"]
    scale = np.where(scale == 0, 1, scale)
    mult = np.where(scale > 0, scale, 1.0 / np.abs(scale))
    raw["source_x"] = raw["source_x"] * mult
    raw["source_y"] = raw["source_y"] * mult
    raw["rec_x"] = raw["rec_x"] * mult
    raw["rec_y"] = raw["rec_y"] * mult
    return raw


def build_gathers(raw, min_traces=6):
    """Group traces by shot id, sort each group by offset. Returns a list
    of dicts, one per shot, with traces/offsets/labels already aligned.

    Offset handling: if the file has a stored OFFSET field, use it - don't
    recompute distance from raw coordinates. On real data checked so far,
    OFFSET matches the true source-receiver distance to well under a meter
    on 3 of 4 assets, but computing distance from raw coordinates gives a
    wildly wrong number on at least one asset (SOURCE_X/REC_X there are
    stored ~100x too large - a COORD_SCALE that never got applied). Rather
    than trust COORD_SCALE blindly (it could be wrong or missing in other
    ways too), we prefer OFFSET when it's available and only fall back to
    the coordinate-based distance - with COORD_SCALE applied - if OFFSET is
    missing.
    """
    samp_rate_us = raw["samp_rate"]
    if np.ptp(samp_rate_us) != 0:
        print("warning: SAMP_RATE isn't constant across traces, using the first value")
    ms_per_sample = float(samp_rate_us[0]) / 1000.0

    if "offset_hdr" in raw:
        # OFFSET already checks out against true distance on every asset
        # seen so far - no need to touch coordinates or COORD_SCALE at all,
        # which also sidesteps asset-specific scale weirdness entirely
        offsets = raw["offset_hdr"]
    else:
        raw = apply_coord_scale(raw)
        offsets = np.sqrt((raw["rec_x"] - raw["source_x"]) ** 2 +
                           (raw["rec_y"] - raw["source_y"]) ** 2)
        print("note: no OFFSET field in this file, using coordinate-based "
              "distance (with COORD_SCALE applied) instead - worth spot "
              "checking a few values against expectation")

    gathers = []
    for sid in np.unique(raw["shot_id"]):
        idx = np.where(raw["shot_id"] == sid)[0]
        if len(idx) < min_traces:
            continue
        order = idx[np.argsort(offsets[idx])]

        fb_ms = raw["fb_ms"][order]
        valid = fb_ms > 0  # 0 or -1 means unlabeled, per the task description
        fb_sample = np.where(valid, np.round(fb_ms / ms_per_sample), -1).astype(int)

        gathers.append({
            "shot_id": sid,
            "traces": raw["data"][order],
            "offsets": offsets[order],
            "fb_ms": fb_ms,
            "fb_sample": fb_sample,
            "valid": valid,
            "ms_per_sample": ms_per_sample,
            "n_labeled": int(valid.sum()),
        })
    return gathers


if __name__ == "__main__":
    import sys
    raw = load_raw(sys.argv[1])
    gathers = build_gathers(raw)
    n_labeled = sum(g["valid"].sum() for g in gathers)
    n_total = sum(len(g["traces"]) for g in gathers)
    print(f"{len(gathers)} shot gathers, {n_total} traces total, {n_labeled} labeled ({100*n_labeled/n_total:.1f}%)")
