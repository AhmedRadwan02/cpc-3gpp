# Configuration Files

Four config files, one per model type. Edit parameters directly in each file.

## Files

1. **`baseline.yaml`** - Baseline autoencoder (no CPC)
2. **`beforeComp.yaml`** - CPC before compression
3. **`afterComp_v1.yaml`** - CPC after compression (V1)
4. **`afterComp_v2.yaml`** - CPC after compression with extra encoder (V2)

## Usage

```bash
# Edit the config you want, then from the repo root:
python -m src.main --config configs/baseline.yaml --run 1
```

Dataset files are expected under `data.path` (default `Dataset/` in the repo YAML). You can override with `--data_path` or the `DATA_ROOT` environment variable when invoking `python -m src.main`.

## Key Parameters to Modify

### Temporal Module
```yaml
temporal_module:
  type: "gru"           # Change to: gru, rnn, lstm, transformer
  num_layers: 1         # Change to: 1, 2, 3
  num_heads: 4          # For transformer: 4, 8, 16
```

### Loss Function
```yaml
contrastive:
  type: "infonce"       # Change to: infonce, simclr, vicreg, none
  temperature: 0.05     # Change to: 0.01, 0.05, 0.1, 0.5
```

### Prediction Steps
```yaml
prediction:
  num_steps: 5          # Change to: 5, 10, 15, 20
```

### Learning Rate
```yaml
optimizer:
  lr: 1.0e-4            # Change to: 1e-3, 1e-4, 1e-5
```

### Batch Size
```yaml
training:
  batch_size: 256       # Change to: 64, 128, 256, 512
```

## Important Notes

- **DO NOT change `model.type`** - it determines which model runs
- **`window_size`**: Use `10` for beforeComp, `null` for others
- **`reconstruction.type`**: Use `sgcs` for beforeComp, `v2_sgcs` for others
- All options are documented inline with comments

## Quick Examples

### Try Transformer instead of GRU
```yaml
# In beforeComp.yaml
temporal_module:
  type: "transformer"
  num_heads: 8
  feedforward_dim: 512
```

### Try SimCLR loss
```yaml
# In beforeComp.yaml
contrastive:
  type: "simclr"
  temperature: 0.5
```

### Increase prediction steps
```yaml
# In any CPC model
prediction:
  num_steps: 10
```

### Try different learning rate
```yaml
# In any config
optimizer:
  lr: 5.0e-5
```
