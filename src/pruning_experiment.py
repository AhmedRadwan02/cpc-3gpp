"""
GRU pruning experiment for beforeComp models.

Uses experiment layout: experiments/{exp_name}_run{N}/{DATASET}/encoder_best.pth, decoder_best.pth.
Example: experiments/beforeComp_gru_sgcs_infonce_temp0.1_pred5_win10_bottleneck32_2bit_run1/CAT/encoder_best.pth
"""

import sys
from pathlib import Path

# Ensure project root is on path when run as script
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import os
import random
import torch
import numpy as np
import pandas as pd
# Use non-interactive backend when no display (e.g. sbatch) so savefig() writes a valid image
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from .models import build_model
from .models.blocks import QuantizedGRU
from .data_loader import load_all_datasets
from .utils.config_utils import load_config
from .trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility (same as main.py before load_all_datasets)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class GRUPruning:
    """Magnitude-based pruning for GRU weights."""

    def __init__(self, module: torch.nn.Module, amount: float = 0.3):
        self.module = module
        self.amount = amount

    def apply(self) -> None:
        for name, param in self.module.named_parameters():
            if "weight" not in name:
                continue
            weight_abs = param.abs().detach()
            n_elements = param.numel()
            n_prune = int(self.amount * n_elements)
            if n_prune == 0:
                continue
            threshold = torch.kthvalue(weight_abs.view(-1), k=n_prune).values
            mask = weight_abs > threshold
            param.data.mul_(mask.float())
            zero_params = (param.data == 0).sum().item()
            sparsity = zero_params / n_elements
            print(f"    {name}: sparsity {sparsity:.2%}")


def prune_model_gru_only(model: torch.nn.Module, pruning_amount: float) -> torch.nn.Module:
    """Prune only QuantizedGRU layers in the model (encoder)."""
    encoder = model.encoder if hasattr(model, "encoder") else model
    for name, module in encoder.named_modules():
        if isinstance(module, QuantizedGRU):
            print(f"  Pruning GRU: {name}")
            pruner = GRUPruning(module, amount=pruning_amount)
            pruner.apply()
    return model


def count_gru_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total GRU params, nonzero GRU params)."""
    encoder = model.encoder if hasattr(model, "encoder") else model
    gru_total, gru_nonzero = 0, 0
    for name, module in encoder.named_modules():
        if isinstance(module, QuantizedGRU):
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    total = param.numel()
                    nonzero = torch.count_nonzero(param).item()
                    gru_total += total
                    gru_nonzero += nonzero
    return gru_total, gru_nonzero


# ---------------------------------------------------------------------------
# Evaluation: use Trainer.test() so pruning ratio=0 matches main exactly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_multi_model_results(
    all_results: Dict[str, List[Dict]],
    save_path: Path,
) -> plt.Figure:
    import matplotlib.ticker

    DATASETS = ['NOKIA', 'OPPO', 'CAT', 'Mixed']
    COLORS  = {'NOKIA': '#1a6faf', 'OPPO': '#d62728', 'CAT': '#2ca02c', 'Mixed': '#ff7f0e'}
    MARKERS = {'NOKIA': 'o',       'OPPO': 's',        'CAT': 'v',       'Mixed': '^'}

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.fontsize':   9,
        'axes.grid':         True,
        'grid.color':        '#cccccc',
        'grid.linestyle':    '--',
        'grid.linewidth':    0.6,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.9,
        'figure.dpi':        300,
        'savefig.dpi':       300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for ds in DATASETS:
        results = all_results.get(ds, [])
        if not results:
            continue
        ratios   = [d['pruning_ratio'] for d in results]
        sgcs     = [d['sgcs_metric']   for d in results]
        info_nce = [d['info_nce_loss'] for d in results]
        ax1.plot(ratios, sgcs,     color=COLORS[ds], marker=MARKERS[ds],
                 markersize=6, linewidth=2.0, label=ds, zorder=3)
        ax2.plot(ratios, info_nce, color=COLORS[ds], marker=MARKERS[ds],
                 markersize=6, linewidth=2.0, label=ds, zorder=3)

    for ax, ylabel, title in [
        (ax1, 'SGCS',         'Model Performance vs Pruning Ratio'),
        (ax2, 'InfoNCE Loss', 'InfoNCE Loss vs Pruning Ratio'),
    ]:
        ax.set_xlabel('Pruning Ratio', labelpad=5)
        ax.set_ylabel(ylabel, labelpad=5)
        ax.set_title(title, fontsize=11, pad=8)
        ax.tick_params(axis='both', which='major', length=4, width=0.8)
        ax.set_facecolor('white')
        ax.legend(loc='best', frameon=True, framealpha=0.95,
                  edgecolor='#bbbbbb', handlelength=2.2,
                  borderpad=0.6, labelspacing=0.5)

    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    fig.tight_layout(pad=1.5)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / 'pruning_analysis_results.pdf', bbox_inches='tight')
    fig.savefig(save_path / 'pruning_analysis_results.png', bbox_inches='tight', dpi=300)
    print(f"Saved figures to {save_path}")
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Experiment name *without* _runN; must match folder experiments/{this}_run1/, ..._run2/, etc.
# Using run1-only by default so ratio=0 SGCS matches main run 1 exactly.
DEFAULT_EXPERIMENT_NAME = "beforeComp_gru_sgcs_infonce_temp0.1_pred5_win10_bottleneck32_hidden128_2bit"
DATASETS = ["CAT", "NOKIA", "OPPO", "Mixed"]
NUM_RUNS = 1


def main(
    base_dir: Path,
    config_path: Path,
    data_path: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    num_runs: int = NUM_RUNS,
    pruning_ratios: np.ndarray = None,
    hidden_size_override: int = None,
    compressed_size_override: int = None,
    pred_steps_override: int = None,
    time_window_override: int = None,
    num_bits_override: int = None,
    num_layers_override: int = None,
    temporal_type_override: str = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve base_dir so we always load from the same place regardless of cwd (e.g. when run via sbatch)
    base_dir = Path(base_dir).resolve()
    print(f"Experiments base_dir (resolved): {base_dir}")

    # Load config (for data, loss, dataset list; architecture can be overridden below)
    experiment_run_dir = base_dir / f"{experiment_name}_run1"
    config_yaml = experiment_run_dir / "config.yaml"
    if config_yaml.exists():
        config = load_config(str(config_yaml))
        print(f"Using experiment config: {config_yaml}")
    else:
        config = load_config(str(config_path))
        print(f"Using config: {config_path}")

    # Architecture: overrides win so you can match checkpoints from ablation regardless of saved config
    compressed_size = compressed_size_override if compressed_size_override is not None else config.model.architecture.compressed_size
    hidden_size = hidden_size_override if hidden_size_override is not None else config.model.architecture.hidden_size
    num_bits = num_bits_override if num_bits_override is not None else config.model.architecture.num_bits
    temporal_type = temporal_type_override if temporal_type_override is not None else config.model.architecture.temporal_module.type
    pred_steps = pred_steps_override if pred_steps_override is not None else config.model.architecture.prediction.num_steps
    time_window = time_window_override if time_window_override is not None else config.model.architecture.temporal_module.time_window
    num_layers = num_layers_override if num_layers_override is not None else config.model.architecture.temporal_module.num_layers

    overrides = []
    if hidden_size_override is not None: overrides.append(f"hidden_size={hidden_size}")
    if compressed_size_override is not None: overrides.append(f"compressed_size={compressed_size}")
    if pred_steps_override is not None: overrides.append(f"pred_steps={pred_steps}")
    if time_window_override is not None: overrides.append(f"time_window={time_window}")
    if num_bits_override is not None: overrides.append(f"num_bits={num_bits}")
    if num_layers_override is not None: overrides.append(f"num_layers={num_layers}")
    if temporal_type_override is not None: overrides.append(f"temporal_type={temporal_type}")
    if overrides:
        print("Overrides (must match checkpoints): " + ", ".join(overrides))

    # Use the exact data path that was used when the run produced the checkpoints (recorded by main).
    data_path_used_file = experiment_run_dir / "data_path_used.txt"
    if data_path_used_file.exists():
        data_path_used = Path(data_path_used_file.read_text().strip())
        if data_path_used.exists():
            data_path = data_path_used
            print("Data path (from data_path_used.txt, same as training):", data_path)
        else:
            data_path_used = None
    else:
        data_path_used = None
    if data_path_used is None:
        data_path_cfg = Path(config.data.path).resolve()
        if data_path_cfg.exists():
            data_path = data_path_cfg
            print("Data path (experiment config, exists):", data_path)
        elif data_path is not None and Path(data_path).resolve().exists():
            data_path = Path(data_path).resolve()
            print("Data path (--data_path override, config path not found):", data_path)
        else:
            data_path = (base_dir.parent / Path(config.data.path).name).resolve()
            print("Data path (fallback, project-relative):", data_path)
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Data path does not exist: {data_path}. "
                    "Use --data_path /path/to/Dataset so pruning uses the same data as the run that produced the checkpoints."
                )
            print("  WARNING: Experiment config path not found; results may not match main unless this is the same data.")

    # Seed before loading data so Mixed split (torch.randperm) matches main. Use run 1's seed
    # since we load config from run1; ensures same train/val/test split as main run 1.
    set_seed(config.experiment.seed + 1)

    # Data
    datasets_dict = load_all_datasets(
        data_path=str(data_path),
        datasets_config=[
            {"name": ds.name, "path": ds.path, "num_samples": ds.num_samples}
            for ds in config.data.datasets
        ],
        window_size=config.data.window_size,
        mixed_samples=config.data.mixed_samples_per_dataset,
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )

    if pruning_ratios is None:
        pruning_ratios = np.arange(0, 1.0, 0.05)

    all_results: Dict[str, List[Dict]] = {ds: [] for ds in DATASETS}

    for dataset_name in DATASETS:
        if dataset_name not in datasets_dict:
            continue
        _, _, test_loader = datasets_dict[dataset_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        run_results: List[List[Dict]] = [[] for _ in range(num_runs)]

        for ratio in pruning_ratios:
            ratio_agg = {
                "gru_total_params": 0,
                "gru_nonzero_params": 0,
                "sgcs_metric": 0.0,
                "info_nce_loss": 0.0,
                "total_loss": 0.0,
            }
            valid_runs = 0

            for run in range(1, num_runs + 1):
                run_dir = base_dir / f"{experiment_name}_run{run}" / dataset_name
                encoder_path = run_dir / "encoder_best.pth"
                decoder_path = run_dir / "decoder_best.pth"
                if not encoder_path.exists() or not decoder_path.exists():
                    print(f"  Skip run{run} (missing {encoder_path} or {decoder_path})")
                    continue

                if ratio == 0 and run == 1:
                    print(f"  [ratio=0 run=1] encoder: {encoder_path}")
                    print(f"  [ratio=0 run=1] decoder: {decoder_path}")

                try:
                    model = build_model(
                        model_type="beforeComp",
                        compressed_size=compressed_size,
                        hidden_size=hidden_size,
                        num_bits=num_bits,
                        temporal_type=temporal_type,
                        num_layers=num_layers,
                        dropout=config.model.architecture.temporal_module.dropout,
                        num_heads=config.model.architecture.temporal_module.num_heads,
                        feedforward_dim=config.model.architecture.temporal_module.feedforward_dim,
                        pred_steps=pred_steps,
                        time_window=time_window,
                    )
                    # Load best checkpoint (same as trainer._save_checkpoint: encoder_best.pth, decoder_best.pth)
                    enc_sd = torch.load(encoder_path, map_location=device)
                    dec_sd = torch.load(decoder_path, map_location=device)
                    model.encoder.load_state_dict(enc_sd, strict=True)
                    model.decoder.load_state_dict(dec_sd, strict=True)

                    if ratio > 0:
                        prune_model_gru_only(model, ratio)

                    model = model.to(device)
                    model.eval()  # match trainer.test(): no dropout, deterministic forward

                    gru_total, gru_nonzero = count_gru_parameters(model)

                    # Use same evaluation as main: Trainer.test() so results match exactly
                    set_seed(config.experiment.seed + run)
                    trainer = Trainer(config, run_dir, device)
                    metrics = trainer.test(model, test_loader)

                    if ratio == 0:
                        test_results_csv = run_dir / "test_results.csv"
                        if test_results_csv.exists():
                            try:
                                main_df = pd.read_csv(test_results_csv)
                                same = (main_df["Train_Dataset"] == dataset_name) & (main_df["Test_Dataset"] == dataset_name)
                                if same.any():
                                    expected_sgcs = float(main_df.loc[same, "SGCS"].iloc[0])
                                    got_sgcs = metrics["sgcs"]
                                    print(f"  [ratio=0] Expected SGCS (main test_results.csv): {expected_sgcs:.6f}, Got: {got_sgcs:.6f}")
                                    if abs(expected_sgcs - got_sgcs) > 1e-4:
                                        print("  WARNING: Mismatch! Check data path and experiment path.")
                            except Exception as e:
                                print(f"  [ratio=0] Could not read expected SGCS from {test_results_csv}: {e}")

                    rec = {
                        "pruning_ratio": ratio,
                        "gru_total_params": gru_total,
                        "gru_nonzero_params": gru_nonzero,
                        "sgcs_metric": metrics["sgcs"],
                        "info_nce_loss": metrics.get("contrast_loss", 0.0),
                        "total_loss": metrics["loss"],
                    }
                    run_results[run - 1].append(rec)
                    ratio_agg["gru_total_params"] += gru_total
                    ratio_agg["gru_nonzero_params"] += gru_nonzero
                    ratio_agg["sgcs_metric"] += metrics["sgcs"]
                    ratio_agg["info_nce_loss"] += metrics.get("contrast_loss", 0.0)
                    ratio_agg["total_loss"] += metrics["loss"]
                    valid_runs += 1
                except Exception as e:
                    print(f"  Run {run} ratio {ratio}: {e}")
                    continue

            if valid_runs == 0:
                continue
            for k in ratio_agg:
                ratio_agg[k] /= valid_runs
            ratio_agg["pruning_ratio"] = ratio
            all_results[dataset_name].append(ratio_agg)
            if ratio in (0, 0.3, 0.5, 0.7):
                print(f"  Ratio {ratio}: SGCS={ratio_agg['sgcs_metric']:.4f} InfoNCE={ratio_agg['info_nce_loss']:.4f}")

    # Save outputs to a dedicated pruning subdir
    output_dir = base_dir / f"pruning_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, results in all_results.items():
        if not results:
            continue
        df = pd.DataFrame(results)
        csv_path = output_dir / f"pruning_{dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    # Combined CSV
    rows = []
    for dataset_name, results in all_results.items():
        for r in results:
            row = r.copy()
            row["model"] = dataset_name
            rows.append(row)
    if rows:
        combined_path = output_dir / "pruning_all.csv"
        pd.DataFrame(rows).to_csv(combined_path, index=False)
        print(f"Saved {combined_path}")

    # Plot and save (no plt.show() when no display so saved file is not empty)
    fig = plot_multi_model_results(all_results, output_dir)
    plt.close(fig)
    if os.environ.get("DISPLAY"):
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GRU pruning for beforeComp experiments")
    parser.add_argument("--base_dir", type=str, default="./experiments", help="Experiments base dir")
    parser.add_argument("--config", type=str, default="configs/beforeComp.yaml", help="Config YAML")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset path (default from config)")
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help="Experiment name without _runN")
    parser.add_argument("--num_runs", type=int, default=NUM_RUNS, help="Number of runs (1-5)")
    parser.add_argument("--max_ratio", type=float, default=0.95, help="Max pruning ratio")
    parser.add_argument("--ratio_step", type=float, default=0.05, help="Pruning ratio step")
    # Architecture overrides (use when experiment config doesn't match checkpoint; you decide values)
    parser.add_argument("--hidden_size", type=int, default=None, help="Override hidden_size")
    parser.add_argument("--compressed_size", type=int, default=None, help="Override compressed_size (bottleneck)")
    parser.add_argument("--pred_steps", type=int, default=None, help="Override prediction steps")
    parser.add_argument("--time_window", type=int, default=None, help="Override time window")
    parser.add_argument("--num_bits", type=int, default=None, help="Override quantization bits")
    parser.add_argument("--num_layers", type=int, default=None, help="Override temporal num_layers")
    parser.add_argument("--temporal_type", type=str, default=None, choices=["gru", "lstm", "rnn", "transformer"], help="Override temporal module type")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    data_path = args.data_path
    if data_path is None:
        config = load_config(str(config_path))
        data_path = config.data.path
    data_path = Path(data_path)

    ratios = np.arange(0, args.max_ratio + 1e-6, args.ratio_step)

    main(
        base_dir=base_dir,
        config_path=config_path,
        data_path=data_path,
        experiment_name=args.experiment,
        num_runs=args.num_runs,
        pruning_ratios=ratios,
        hidden_size_override=args.hidden_size,
        compressed_size_override=args.compressed_size,
        pred_steps_override=args.pred_steps,
        time_window_override=args.time_window,
        num_bits_override=args.num_bits,
        num_layers_override=args.num_layers,
        temporal_type_override=args.temporal_type,
    )
