"""
Efficiency experiment: compute per-batch metrics (time, FLOPs, params) for a given experiment.

Uses same pattern as pruning_experiment: load config from experiment or YAML, load data via
load_all_datasets, build model (optionally load weights from experiment run), run one batch
through compute_efficiency_metrics_per_batch, save to efficiency_{experiment_name}/.

Usage:
  python -m src.efficiency_experiment --config configs/beforeComp.yaml --experiment beforeComp_gru_sgcs_infonce_temp0.1_pred5_win10_bottleneck32_hidden128_2bit
  python -m src.efficiency_experiment --experiment beforeComp_gru_sgcs_infonce_temp0.1_pred5_win10_bottleneck32_hidden128_2bit --run 1 --dataset Mixed
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any, Optional

from .models import build_model
from .data_loader import load_all_datasets
from .utils.config_utils import load_config
from .utils.efficiency_metrics import compute_efficiency_metrics_per_batch


def get_one_batch(datasets_dict: Dict, dataset_name: str, model_type: str, time_window: Optional[int]) -> torch.Tensor:
    """Get one batch from the dataset's train loader. For beforeComp, ensure enough frames for time_window."""
    train_loader, _, _ = datasets_dict[dataset_name]
    batch = next(iter(train_loader))[0]
    # Flatten 5D -> 4D if needed (beforeComp dataloader may give (B, T, 2, 13, 32))
    if batch.dim() == 5 and model_type == "beforeComp":
        batch = batch.reshape(-1, batch.size(2), batch.size(3), batch.size(4))
    if model_type == "beforeComp" and time_window is not None and batch.size(0) < time_window:
        n = (time_window + batch.size(0) - 1) // batch.size(0)
        batch = batch.repeat(n, 1, 1, 1)[:time_window]
    return batch


def main(
    base_dir: Path,
    config_path: Path,
    data_path: Path,
    experiment_name: str,
    run: int = 1,
    dataset_name: Optional[str] = None,
    load_weights: bool = True,
    hidden_size_override: Optional[int] = None,
    compressed_size_override: Optional[int] = None,
    pred_steps_override: Optional[int] = None,
    time_window_override: Optional[int] = None,
    num_bits_override: Optional[int] = None,
    num_layers_override: Optional[int] = None,
    temporal_type_override: Optional[str] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config from experiment run dir or from config path
    experiment_run_dir = base_dir / f"{experiment_name}_run{run}"
    config_yaml = experiment_run_dir / "config.yaml"
    if config_yaml.exists():
        config = load_config(str(config_yaml))
        print(f"Using experiment config: {config_yaml}")
    else:
        config = load_config(str(config_path))
        print(f"Using config: {config_path}")

    model_type = config.model.type
    compressed_size = compressed_size_override if compressed_size_override is not None else config.model.architecture.compressed_size
    hidden_size = hidden_size_override if hidden_size_override is not None else config.model.architecture.hidden_size
    num_bits = num_bits_override if num_bits_override is not None else config.model.architecture.num_bits
    temporal_type = temporal_type_override or config.model.architecture.temporal_module.type
    pred_steps = pred_steps_override if pred_steps_override is not None else config.model.architecture.prediction.num_steps
    time_window = time_window_override if time_window_override is not None else getattr(
        config.model.architecture.temporal_module, "time_window", 10
    )
    num_layers = num_layers_override if num_layers_override is not None else config.model.architecture.temporal_module.num_layers

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

    # Which dataset to use for the batch (and for loading weights if load_weights)
    if dataset_name is None:
        dataset_name = "Mixed" if "Mixed" in datasets_dict else list(datasets_dict.keys())[0]
    if dataset_name not in datasets_dict:
        raise ValueError(f"Dataset {dataset_name} not in {list(datasets_dict.keys())}")

    sample_batch = get_one_batch(datasets_dict, dataset_name, model_type, time_window if model_type == "beforeComp" else None)
    print(f"Sample batch shape: {sample_batch.shape} (dataset={dataset_name})")

    # Build model
    if model_type == "baseline":
        model = build_model(
            model_type="baseline",
            compressed_size=compressed_size,
            num_bits=num_bits,
        )
    else:
        model = build_model(
            model_type=model_type,
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

    if load_weights:
        run_dir = base_dir / f"{experiment_name}_run{run}" / dataset_name
        encoder_path = run_dir / "encoder_best.pth"
        decoder_path = run_dir / "decoder_best.pth"
        if encoder_path.exists() and decoder_path.exists():
            model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            print(f"Loaded weights from {run_dir}")
        else:
            print(f"Weights not found at {run_dir}; measuring architecture only (random init)")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Per-batch efficiency
    metrics = compute_efficiency_metrics_per_batch(
        model, model_type, sample_batch, device, num_warmup=10, num_repeat=100
    )
    if "error" in metrics:
        print(f"Efficiency error: {metrics['error']}")
    else:
        print(f"Encoder params: {metrics['encoder_params']:,}")
        print(f"Decoder params: {metrics['decoder_params']:,}")
        print(f"Encoder time (ms per batch): {metrics['encoder_time_ms_per_batch']}")
        print(f"Decoder time (ms per batch): {metrics['decoder_time_ms_per_batch']}")
        print(f"Encoder FLOPs (per batch): {metrics.get('encoder_flops_per_batch')}")
        print(f"Decoder FLOPs (per batch): {metrics.get('decoder_flops_per_batch')}")
        print(f"Batch size: {metrics['batch_size']}, num_windows: {metrics.get('num_windows')}")

    # Save
    output_dir = base_dir / f"efficiency_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment": experiment_name,
        "model_type": model_type,
        "dataset": dataset_name,
        "run": run,
        "encoder_params": metrics["encoder_params"],
        "decoder_params": metrics["decoder_params"],
        "encoder_time_ms_per_batch": metrics.get("encoder_time_ms_per_batch"),
        "decoder_time_ms_per_batch": metrics.get("decoder_time_ms_per_batch"),
        "encoder_flops_per_batch": metrics.get("encoder_flops_per_batch"),
        "decoder_flops_per_batch": metrics.get("decoder_flops_per_batch"),
        "batch_size": metrics.get("batch_size"),
        "num_windows": metrics.get("num_windows"),
    }
    if "error" in metrics:
        row["error"] = metrics["error"]

    json_path = output_dir / "efficiency_per_batch.json"
    with open(json_path, "w") as f:
        json.dump({**metrics, "experiment": experiment_name, "dataset": dataset_name, "run": run}, f, indent=2)
    print(f"Saved {json_path}")

    csv_path = output_dir / "efficiency_per_batch.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiency (per-batch) for an experiment")
    parser.add_argument("--base_dir", type=str, default="./experiments", help="Experiments base dir")
    parser.add_argument("--config", type=str, default="configs/beforeComp.yaml", help="Config YAML (used if experiment config missing)")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset path (default from config)")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name without _runN (e.g. beforeComp_gru_sgcs_infonce_temp0.1_pred5_win10_bottleneck32_hidden128_2bit)")
    parser.add_argument("--run", type=int, default=1, help="Run number to load config/weights from")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset for batch and weights (default: Mixed or first)")
    parser.add_argument("--no_weights", action="store_true", help="Do not load weights; measure architecture only")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--compressed_size", type=int, default=None)
    parser.add_argument("--pred_steps", type=int, default=None)
    parser.add_argument("--time_window", type=int, default=None)
    parser.add_argument("--num_bits", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--temporal_type", type=str, default=None, choices=["gru", "lstm", "rnn", "transformer"])
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

    main(
        base_dir=base_dir,
        config_path=config_path,
        data_path=data_path,
        experiment_name=args.experiment,
        run=args.run,
        dataset_name=args.dataset,
        load_weights=not args.no_weights,
        hidden_size_override=args.hidden_size,
        compressed_size_override=args.compressed_size,
        pred_steps_override=args.pred_steps,
        time_window_override=args.time_window,
        num_bits_override=args.num_bits,
        num_layers_override=args.num_layers,
        temporal_type_override=args.temporal_type,
    )
