# Scripts - Simple Usage

Four scripts, one per model type. Just sbatch the one you want.

## Files

```
scripts/
├── run_baseline.sh        # Baseline model
├── run_beforeComp.sh      # BeforeComp model
├── run_afterComp_v1.sh    # AfterComp V1 model
└── run_afterComp_v2.sh    # AfterComp V2 model
```

## Usage

### Basic (Run 1, all datasets)
```bash
sbatch scripts/run_baseline.sh
sbatch scripts/run_beforeComp.sh
sbatch scripts/run_afterComp_v1.sh
sbatch scripts/run_afterComp_v2.sh
```

### Specify Run Number
```bash
sbatch scripts/run_baseline.sh 2
sbatch scripts/run_beforeComp.sh 3
```

### Train on Single Dataset (faster)
```bash
sbatch scripts/run_baseline.sh 1 NOKIA
sbatch scripts/run_beforeComp.sh 1 OPPO
sbatch scripts/run_afterComp_v1.sh 1 CAT
sbatch scripts/run_afterComp_v2.sh 1 Mixed
```

## Workflow

1. **Edit config** (if needed):
   ```bash
   vim configs/baseline.yaml
   ```

2. **Submit job**:
   ```bash
   sbatch scripts/run_baseline.sh
   ```

3. **Monitor**:
   ```bash
   squeue -u $USER
   tail -f logs/baseline_*.out
   ```

4. **Check results**:
   ```bash
   ls experiments/
   cat experiments/NOKIA_baseline_2bit_run1/test_results.csv
   ```

## Examples

```bash
# Baseline run 1, all datasets
sbatch scripts/run_baseline.sh

# BeforeComp run 2, NOKIA only
sbatch scripts/run_beforeComp.sh 2 NOKIA

# AfterComp V1 run 3, all datasets
sbatch scripts/run_afterComp_v1.sh 3

# AfterComp V2 run 1, Mixed only
sbatch scripts/run_afterComp_v2.sh 1 Mixed
```

## Logs

Logs are saved as:
- `logs/baseline_JOBID.out`
- `logs/beforeComp_JOBID.out`
- `logs/afterComp_v1_JOBID.out`
- `logs/afterComp_v2_JOBID.out`

## That's It!

Just 4 scripts. Edit configs, sbatch the script you want. Simple! 👍
