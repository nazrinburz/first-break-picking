# Seismic First-Break Picking

This repository contains the data processing, training, evaluation, and inference pipeline for automatic seismic first-break picking across four real-world hard-rock mining assets:

- Brunswick
- Halfmile
- Lalor
- Sudbury

## Approaches

The project evolved through three approaches:

1. **Classical baseline — STA/LTA + AIC**  
   Detects energy changes using STA/LTA and refines the first-break location using AIC.

2. **1D CNN — Historical**  
   Predicts first-break time from an individual seismic trace and its offset. Historical leave-one-asset-out experiments showed substantial cross-asset variation.

3. **2D CNN — Final approach**  
   `Gather2DCNN` processes 64 neighboring traces together, allowing the model to use spatial continuity and moveout information across the shot gather.

The final reported result is an **in-domain, shot-level test MAE of 13.20 ms** on **56,195 labeled traces**.

## Project Structure

```text
├── data/
│   ├── Brunswick_orig_1500ms_V2.hdf5
│   ├── Halfmile3D_add_geom_sorted.hdf5
│   ├── Lalor_raw_z_1500ms_norp_geom_v3.hdf5
│   └── preprocessed_Sudbury3D.hdf
│
├── Final 2D CNN (Primary)
│   ├── train_2d.py              # Train the final 2D CNN
│   ├── model_2d.py              # Gather2DCNN architecture
│   ├── gather_dataset_2d.py     # 2D gather dataset and windowing
│   └── inference_2d.py          # Single-shot inference
│
├── Historical 1D CNN & Multi-Asset Training
│   ├── train.py                 # Train the historical 1D CNN
│   ├── model.py                 # 1D CNN architecture
│   ├── dataset.py               # 1D trace dataset loading
│   ├── train_multi.py           # Multi-asset model training pipeline
│   └── multi_dataset.py         # Multi-asset dataset handling
│
└── Baselines & Utilities
    ├── baseline.py              # Classical STA/LTA + AIC baseline approach
    ├── data_utils.py            # HDF5 loading and gather construction
    ├── resample.py              # Trace resampling
    ├── evaluate.py              # Evaluation utilities
    ├── report.py                # Training reports
    ├── seed_utils.py            # Reproducibility
    ├── check_data.py            # Data validation and exploration
    ├── make_synthetic_data.py   # Synthetic data generation
    ├── run_demo.py              # Demonstration script
    ├── visualize.py             # Plotting and visualization tools
    └── analyze.ipynb            # Jupyter notebook for data analysis
```

> **Note**: Historical 1D experiments, baseline results and LLO results are retained in `1cnn/`, `findings/` and `loo_results_1d/`, respectively.

## Data Processing

Raw HDF5 traces are:

- Grouped by `SHOTID`
- Sorted by `OFFSET`
- Resampled to 2 ms × 750 samples (~1500 ms)
- Normalized using the 99.5th percentile of absolute amplitude
- Clipped to `[-3, 3]`
- Split into windows of 64 traces

The final 2D input has shape:

```text
[B, 3, 64, 750]
```

The three channels are:

1. Normalized seismic waveform
2. Normalized source-receiver offset
3. Trace/padding mask

Unlabeled real traces are retained as spatial context. A separate label mask ensures that only labeled traces contribute to the supervised loss.

## Model

`Gather2DCNN` uses 2D convolutions across neighboring traces and time.

```text
Input:  [B, 3, 64, 750]
            ↓
        2D CNN
            ↓
Output: [B, 64]
```

The trace dimension is preserved throughout the network, producing one first-break time prediction per trace.

## Evaluation

The final benchmark uses an in-domain shot-level split:

| Asset | Train | Validation | Test |
| :--- | :--- | :--- | :--- |
| Brunswick | 75 | 15 | 10 |
| Halfmile | 75 | 15 | 10 |
| Lalor | 75 | 15 | 10 |
| Sudbury | 50 | 9 | 6 |

Splitting is performed by shot before creating windows to avoid trace-level leakage.

### Validation

- **75,964 labeled traces**
- **MAE:** 16.09 ms
- **Median absolute error:** 8.42 ms
- **Within ±4 ms:** 26.7%
- **Within ±8 ms:** 48.2%
- **Within ±16 ms:** 72.3%

### Test

- **56,195 labeled traces**
- **MAE:** 13.20 ms
- **Median absolute error:** 7.69 ms
- **Within ±4 ms:** 28.6%
- **Within ±8 ms:** 51.5%
- **Within ±16 ms:** 76.2%

## Training

The final model was trained with:

```bash
python -u train_2d.py \
  --assets brunswick=data/Brunswick_orig_1500ms_V2.hdf5 \
           halfmile=data/Halfmile3D_add_geom_sorted.hdf5 \
           lalor=data/Lalor_raw_z_1500ms_norp_geom_v3.hdf5 \
           sudbury=data/preprocessed_Sudbury3D.hdf \
  --mode in_domain \
  --max_shots 100 \
  --epochs 20 \
  --patience 5 \
  --batch_size 32 \
  --max_traces 64 \
  --trace_stride 64 \
  --seed 42 \
  --target_dt_ms 2.0 \
  --target_n_samples 750 \
  --report_prefix fbpick_2d_final
```

The best checkpoint was selected using validation MAE and restored before saving.

## Inference

Run inference on a specific shot:

```bash
python inference_2d.py \
  --checkpoint fbpick_2d_final_model.pt \
  --asset brunswick \
  --data data/Brunswick_orig_1500ms_V2.hdf5 \
  --shot_id 1152 \
  --output inference_brunswick_test0.png
```

Brunswick `SHOTID 1152` is a confirmed held-out test shot:

- 3,299 traces
- 2,649 labeled traces
- MAE: 14.09 ms
- Median absolute error: 7.86 ms

List available test shots:

```bash
python inference_2d.py \
  --checkpoint fbpick_2d_final_model.pt \
  --asset brunswick \
  --data data/Brunswick_orig_1500ms_V2.hdf5 \
  --list_test_shots
```

Automatically select a held-out test shot:

```bash
python inference_2d.py \
  --checkpoint fbpick_2d_final_model.pt \
  --asset brunswick \
  --data data/Brunswick_orig_1500ms_V2.hdf5 \
  --use_test_shot
```

## Historical 1D Leave-One-Asset-Out Results

The historical 1D CNN was evaluated by training on three assets and testing on the fourth:

| Held-out asset | MAE |
| :--- | :--- |
| Brunswick | 108.64 ms |
| Halfmile | 131.21 ms |
| Lalor | 20.73 ms |
| Sudbury | 18.34 ms |

*Note: These results use a different evaluation protocol from the final 2D benchmark and are therefore not directly comparable with the 13.20 ms result.*

## Limitations

The final 2D benchmark is in-domain. Cross-asset generalization has not yet been reported for the final 2D model. 

Other limitations include:

- Sparse labels in Sudbury
- 64-trace local windows rather than full gathers
- Geometry represented primarily by scalar offset
- Lalor downsampled from 1 ms to 2 ms
- Potential spatial correlation between nearby shots

A strict leave-one-asset-out evaluation of the final 2D CNN is the main next step.