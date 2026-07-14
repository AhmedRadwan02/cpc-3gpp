# Contrastive Predictive Coding with Compression for Enhanced Channel State Feedback in Wireless Networks

**Ahmed Y. Radwan** and **Hina Tabassum** — Department of Electrical Engineering and Computer Science, York University, Toronto, ON, Canada.
Fahad Syed Muhammad - Nokia France.
Matthew Baker - Nokia UK.

The codebase is **config-driven**: it compresses **3GPP-style Channel State Information (CSI)** with neural autoencoders, combining reconstruction objectives (including SGCS-style losses) with **Contrastive Predictive Coding (CPC)** so temporal structure in CSI sequences is used in line with CSI feedback settings. Variants include a baseline quantized autoencoder, CPC **before** the compression bottleneck, and two CPC-after-reconstruction models (V1/V2). Training and evaluation are run with **Python** from the repo root, using YAML under `configs/` and `python -m src.main`.

---

## Quick Start

### 1. Environment

From the repository root:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Check the install

```bash
python test_setup.py
```

You should see checks for config loading, models, and related components ending with all tests passed.

### 3. First training run (local)

```bash
python -m src.main --config configs/baseline.yaml --run 1
```

Other model configs: `configs/beforeComp.yaml`, `configs/afterComp_v1.yaml`, `configs/afterComp_v2.yaml`. Train on one vendor only, e.g. `python -m src.main --config configs/baseline.yaml --run 1 --train_dataset NOKIA`.

Optional dataset root override (defaults to `Dataset/` in the YAML):

```bash
python -m src.main --config configs/baseline.yaml --run 1 --data_path /path/to/Dataset
# or: export DATA_ROOT=/path/to/Dataset
```

### 4. Multiple runs (optional)

From the repo root, repeat the same config with different run indices (for example runs 1–5):

```bash
for r in 1 2 3 4 5; do
  python -m src.main --config configs/baseline.yaml --run "$r"
done
```

### 5. Outputs

Each run writes under `experiments/` (not tracked in git): `config.yaml`, checkpoints, and `test_results.csv` inside the run folder.

---

## Dataset

The CSI dataset used in this work is the official 3GPP NR AI/ML dataset for CSI compression, published by 3GPP RAN WG4.

**Download:** [3GPP R4-113 CSI Compression Datasets](https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/Data_sharing/NR_AIML_air/CSI_compression/Datasets/R4_113)

After downloading, place the vendor folders under `Dataset/` in the repo root:

```
Dataset/
├── Nokia/
├── OPPO/
└── CAT/
```

Then run as normal — configs default to `data.path: "Dataset"`, or override with `--data_path` or `DATA_ROOT`.

---

## Features

- **Config-driven experiments**: All hyperparameters defined in YAML files
- **Multiple model variants**: Baseline, CPC before/after compression (V1/V2)
- **Flexible temporal modules**: GRU, RNN, LSTM, Transformer
- **Multiple loss functions**: InfoNCE, SimCLR, VICReg, SGCS
- **Quantization**: 2-bit weight quantization with straight-through estimators
- **Multi-GPU support**: Automatic DataParallel wrapping

---

## Project Structure

```
3GPP_Article/
├── configs/                 # YAML experiment definitions (see configs/README.md)
│   ├── baseline.yaml
│   ├── beforeComp.yaml
│   ├── afterComp_v1.yaml
│   └── afterComp_v2.yaml
├── Dataset/                 # CSI .npy inputs — not tracked in git, see Dataset section above
│   ├── Nokia/
│   ├── OPPO/
│   └── CAT/
├── src/                     # Library and training entry point
│   ├── models/              # Architectures (baseline, before/after CPC)
│   ├── utils/               # Config helpers, metrics
│   ├── losses.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── main.py              # python -m src.main ...
├── experiments/             # Ignored: checkpoints and logs from training
└── requirements.txt
```

---

## Model Variants

### 1. Baseline
Simple autoencoder without CPC:
- Encoder: Conv → 6 ResNet blocks → Compress to 32 dims
- Decoder: Expand → 6 ResNet blocks → Conv
- Loss: SGCS only

### 2. BeforeComp
CPC applied **before** compression bottleneck:
- Encoder: Conv → ResNet → **Temporal Module → Predictions** → Compress to 32 dims
- Decoder: Expand → Residual blocks
- Loss: SGCS + Contrastive (InfoNCE/SimCLR/VICReg)

### 3. AfterComp V1
CPC applied **after** reconstruction:
- Encoder: Conv → 6 ResNet blocks → Compress to 32 dims
- Decoder: Expand → 6 ResNet blocks → Reconstruct → **Temporal Module → Predictions**
- Loss: SGCS (reconstruction) + Contrastive (predictions)

### 4. AfterComp V2
CPC after reconstruction with extra encoder:
- Same as V1, but adds extra conv encoder before temporal module
- Decoder: ... → Reconstruct → **Extra Encoder** → Temporal Module → Predictions

---

## Usage

Always run these commands from the **repository root** so `src` and `configs/` resolve correctly.

```bash
# One training run (baseline)
python -m src.main --config configs/baseline.yaml --run 1

# Other model configs
python -m src.main --config configs/beforeComp.yaml --run 1
python -m src.main --config configs/afterComp_v1.yaml --run 1
python -m src.main --config configs/afterComp_v2.yaml --run 1

# Optional: dataset root (defaults to Dataset/ from the YAML)
python -m src.main --config configs/baseline.yaml --run 1 --data_path /path/to/Dataset
# or: export DATA_ROOT=/path/to/Dataset

# Train on one vendor only (NOKIA, OPPO, CAT, or Mixed)
python -m src.main --config configs/baseline.yaml --run 1 --train_dataset NOKIA

# Skip pruning sweep after training (beforeComp only, if applicable)
python -m src.main --config configs/beforeComp.yaml --run 1 --skip_pruning
```

---

## Configuration

Example config structure (`configs/beforeComp.yaml`):

```yaml
experiment:
  name: "beforeComp"
  seed: 42
  num_runs: 5

model:
  type: "beforeComp"  # baseline | beforeComp | afterComp_v1 | afterComp_v2
  architecture:
    compressed_size: 32
    hidden_size: 64
    num_bits: 2
    temporal_module:
      type: "gru"  # gru | rnn | lstm | transformer
      num_layers: 1
      dropout: 0.0
      num_heads: 4  # For transformer
      feedforward_dim: 256  # For transformer
    prediction:
      num_steps: 5  # Number of future steps to predict

loss:
  reconstruction:
    type: "sgcs"  # sgcs | v2_sgcs | mse | l1
    weight: 0.5
  contrastive:
    type: "infonce"  # infonce | simclr | vicreg | none
    weight: 0.5
    temperature: 0.05  # For InfoNCE/SimCLR
    # VICReg coefficients
    sim_coeff: 25.0
    std_coeff: 25.0
    cov_coeff: 1.0

training:
  optimizer:
    type: "adam"
    lr: 1.0e-4
    weight_decay: 0.0
  epochs: 150
  early_stopping:
    patience: 25
    metric: "val_sgcs"
  batch_size: 256

data:
  path: "Dataset"
  datasets:
    - name: "NOKIA"
      path: "Nokia/N0KIR4_113dsei.npy"
      num_samples: 600000
    - name: "OPPO"
      path: "OPPO/OPPOR4_113dsei00.npy"
      num_samples: 600000
    - name: "CAT"
      path: "CAT/CAT0R4_113dsei01.npy"
      num_samples: 100000
  window_size: 10  # For CPC models, null for baseline
```

---

## Experiment Outputs

Each experiment creates a folder with descriptive name:

```
experiments/
├── NOKIA_baseline_2bit_run1/
│   ├── config.yaml              # Copy of config used
│   ├── encoder_best.pth         # Best encoder weights
│   ├── decoder_best.pth         # Best decoder weights
│   └── test_results.csv         # Cross-dataset evaluation
├── Mixed_beforeComp_gru_infonce_temp0.05_pred5_2bit_run1/
│   └── ...
└── run1_all_results.csv         # Aggregated results
```

---

## Performance Comparison

Based on original results:

| Model | NOKIA→NOKIA | OPPO→OPPO | Mixed→Mixed |
|-------|-------------|-----------|-------------|
| Baseline | 0.7245 | 0.7325 | 0.6929 |
| BeforeComp (GRU+InfoNCE) | **0.9093** | **0.8954** | **0.8765** |
| AfterComp V1 | 0.7271 | 0.7393 | 0.6880 |
| AfterComp V2 | 0.7297 | 0.7329 | 0.6965 |

---

## Extending the Codebase

### New Temporal Module

Edit `src/models/temporal.py`:

```python
class MyTemporalModule(TemporalModule):
    def forward(self, x):
        # Your implementation
        return context, hidden_states

# Register in build_temporal_module()
```

### New Loss Function

Edit `src/losses.py`:

```python
class MyLoss(nn.Module):
    def forward(self, pred, target):
        # Your implementation
        return loss

# Register in build_contrastive_loss()
```

### New Config

Copy an existing file under `configs/`, edit hyperparameters, then run with `python -m src.main --config configs/your_experiment.yaml --run 1`.

---

## Tips

- **Window size**: Use `window_size: 10` for CPC models, `null` for baseline
- **Loss type**: Use `v2_sgcs` for spatial reconstruction (baseline, afterComp), `sgcs` for prediction (beforeComp)
- **Temperature**: Lower (0.05) for InfoNCE, higher (0.5) for SimCLR
- **Prediction steps**: 5–10 works well, more increases compute
- **Batch size**: Automatically multiplied by number of GPUs

---

## Troubleshooting

**"No valid batches"**: Reduce batch size or increase dataset size

**OOM errors**: Reduce batch size, hidden size, or number of layers

**Poor convergence**: Adjust learning rate, try different temporal module

**"ModuleNotFoundError: No module named 'src'"**: Run from the repository root, not from inside `src/`

---

## Citation

```
@article{radwan2026contrastive,
  title={Contrastive Predictive Coding With Compression for Enhanced Channel State Feedback in Wireless Networks},
  author={Radwan, Ahmed Y and Muhammad, Fahad Syed and Baker, Matthew and Tabassum, Hina},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026},
  publisher={IEEE}
}
  ```
