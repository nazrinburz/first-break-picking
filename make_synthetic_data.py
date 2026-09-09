"""
Not part of the actual deliverable - just builds a tiny fake hdf5 with the
same group/dataset layout as the real files, so the pipeline can be sanity
checked without pulling down multi-GB assets first. Swap in the real file
paths once you've downloaded Brunswick / Halfmile / Lalor / Sudbury.
"""
import numpy as np
import h5py


def make_file(path, n_shots=25, n_rec=48, n_samples=1500, dt_ms=1.0, spike_prob=0.02, seed=0):
    rng = np.random.default_rng(seed)

    shot_id, source_x, source_y, rec_x, rec_y = [], [], [], [], []
    data_rows, fb_ms = [], []

    for s in range(n_shots):
        sx, sy = rng.uniform(0, 500), 0.0
        rx = np.sort(rng.uniform(0, 1000, n_rec))
        ry = np.zeros(n_rec)

        offset = np.abs(rx - sx)
        velocity = rng.uniform(1500, 2500)     # m/s, made up for the toy data
        t_ms = 5 + offset / velocity * 1000    # linear moveout + small static

        for r in range(n_rec):
            trace = rng.normal(0, 0.05, n_samples)
            onset = int(round(t_ms[r] / dt_ms))
            if onset < n_samples - 50:
                tail = n_samples - onset
                decay = np.exp(-np.linspace(0, 4, tail))
                # start the wavelet at peak phase instead of zero-crossing so
                # the onset is a sharp kick, like a real first arrival, not a
                # slow fade-in
                wavelet = np.sin(np.linspace(np.pi / 2, 12 * np.pi, tail)) * decay
                trace[onset:] += wavelet * rng.uniform(0.6, 1.4)

            # occasional huge-amplitude outlier sample, like the spikes seen
            # in the real gather plots, to make sure preprocessing handles it
            if rng.random() < spike_prob:
                trace[rng.integers(0, n_samples)] += rng.uniform(20, 50)

            data_rows.append(trace)
            shot_id.append(s)
            source_x.append(sx)
            source_y.append(sy)
            rec_x.append(rx[r])
            rec_y.append(ry[r])
            fb_ms.append(t_ms[r] if rng.random() > 0.05 else -1)  # some unlabeled, like real data

    n = len(data_rows)
    with h5py.File(path, "w") as f:
        g = f.create_group("TRACE_DATA/DEFAULT")
        g.create_dataset("data_array", data=np.array(data_rows, dtype=np.float32))
        g.create_dataset("SHOTID", data=np.array(shot_id))
        g.create_dataset("SOURCE_X", data=np.array(source_x))
        g.create_dataset("SOURCE_Y", data=np.array(source_y))
        g.create_dataset("REC_X", data=np.array(rec_x))
        g.create_dataset("REC_Y", data=np.array(rec_y))
        g.create_dataset("OFFSET", data=np.abs(np.array(rec_x) - np.array(source_x)))
        g.create_dataset("SAMP_RATE", data=np.full(n, int(dt_ms * 1000)))
        g.create_dataset("COORD_SCALE", data=np.ones(n))
        g.create_dataset("SPARE1", data=np.array(fb_ms))


if __name__ == "__main__":
    make_file("synthetic_asset.hdf5")
    print("wrote synthetic_asset.hdf5")
