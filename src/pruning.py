"""
Magnitude-based GRU pruning utilities.

Designed to be called from main.py immediately after training,
using the same test_loader objects already in memory — this guarantees
pruning ratio=0 results match test_results.csv exactly.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from .trainer import Trainer
from .utils.config_utils import Config


# ---------------------------------------------------------------------------
# Core pruning primitives
# ---------------------------------------------------------------------------

class GRUPruning:
    """Unstructured magnitude-based pruning for GRU weights (in-place)."""

    def __init__(self, module: nn.Module, amount: float):
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
            sparsity = (param.data == 0).sum().item() / n_elements
            print(f"    {name}: sparsity {sparsity:.2%}", flush=True)


def prune_model_gru_only(model: nn.Module, pruning_amount: float) -> nn.Module:
    """
    Prune only QuantizedGRU layers inside the encoder. Returns the same model
    (in-place modification), so always pass a freshly loaded copy per ratio.
    """
    # Import here to avoid circular imports at module level
    from .models.blocks import QuantizedGRU

    encoder = model.encoder if hasattr(model, "encoder") else model
    for name, module in encoder.named_modules():
        if isinstance(module, QuantizedGRU):
            print(f"  Pruning GRU layer: {name}", flush=True)
            GRUPruning(module, amount=pruning_amount).apply()
    return model


def count_gru_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_gru_weight_params, nonzero_gru_weight_params)."""
    from .models.blocks import QuantizedGRU

    encoder = model.encoder if hasattr(model, "encoder") else model
    total, nonzero = 0, 0
    for _, module in encoder.named_modules():
        if isinstance(module, QuantizedGRU):
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    total += param.numel()
                    nonzero += torch.count_nonzero(param).item()
    return total, nonzero


# ---------------------------------------------------------------------------
# High-level sweep
# ---------------------------------------------------------------------------

def run_pruning_sweep(
    model: nn.Module,
    trainer: Trainer,
    datasets_dict: dict,
    train_dataset_name: str,
    experiment_dir: Path,
    config: Config,
    device: torch.device,
) -> None:
    """
    Run a magnitude-based GRU pruning sweep and save results to
    experiment_dir/pruning_results.csv.

    Called from main.py right after trainer.train() + the normal test loop,
    so test_loaders are the exact same objects used during training — this
    guarantees ratio=0 SGCS matches test_results.csv.

    Args:
        model:              The best trained model (already loaded, on device).
        trainer:            The Trainer instance (has .test() and config).
        datasets_dict:      {dataset_name: (train_loader, val_loader, test_loader)}
        train_dataset_name: Name of the dataset this model was trained on.
        experiment_dir:     Path to save pruning_results.csv.
        config:             Full config object.
        device:             torch.device.
    """
    # Resolve pruning ratios from config if available, else use default
    if hasattr(config, "pruning") and hasattr(config.pruning, "ratios"):
        pruning_ratios = list(config.pruning.ratios)
    else:
        pruning_ratios = [round(r, 2) for r in np.arange(0.0, 0.95 + 1e-6, 0.05)]

    # Save the best model state once so we can reload for each ratio
    best_state = copy.deepcopy({
        "encoder": (model.module if isinstance(model, nn.DataParallel) else model).encoder.state_dict(),
        "decoder": (model.module if isinstance(model, nn.DataParallel) else model).decoder.state_dict(),
    })

    encoder_path = experiment_dir / "encoder_best.pth"
    decoder_path = experiment_dir / "decoder_best.pth"

    print(f"\n{'='*70}", flush=True)
    print(f"Pruning sweep for model trained on: {train_dataset_name}", flush=True)
    print(f"Ratios: {pruning_ratios}", flush=True)
    print(f"{'='*70}", flush=True)

    all_rows: List[Dict] = []

    for ratio in pruning_ratios:
        print(f"\n--- Pruning ratio: {ratio:.2f} ---", flush=True)

        # --- Build a fresh model and reload best weights for every ratio ---
        # This ensures each ratio prunes from the clean trained model,
        # not from a previously pruned state.
        from .models import build_model

        fresh_model = build_model(
            model_type=config.model.type,
            compressed_size=config.model.architecture.compressed_size,
            hidden_size=config.model.architecture.hidden_size,
            num_bits=config.model.architecture.num_bits,
            temporal_type=config.model.architecture.temporal_module.type,
            num_layers=config.model.architecture.temporal_module.num_layers,
            dropout=config.model.architecture.temporal_module.dropout,
            num_heads=config.model.architecture.temporal_module.num_heads,
            feedforward_dim=config.model.architecture.temporal_module.feedforward_dim,
            pred_steps=config.model.architecture.prediction.num_steps,
            time_window=config.model.architecture.temporal_module.time_window,
        )

        # Load best weights (from in-memory copy — no disk I/O needed)
        fresh_model.encoder.load_state_dict(best_state["encoder"])
        fresh_model.decoder.load_state_dict(best_state["decoder"])

        # Apply pruning (skip for ratio=0 — this IS our ground truth baseline)
        if ratio > 0.0:
            prune_model_gru_only(fresh_model, ratio)

        # Wrap with DataParallel if needed, move to device
        if torch.cuda.device_count() > 1:
            fresh_model = nn.DataParallel(fresh_model)
        fresh_model = fresh_model.to(device)

        # Count GRU sparsity
        gru_total, gru_nonzero = count_gru_parameters(fresh_model)
        actual_sparsity = 1.0 - (gru_nonzero / gru_total) if gru_total > 0 else 0.0
        print(f"  GRU params: {gru_total:,} total, {gru_nonzero:,} nonzero "
              f"(actual sparsity: {actual_sparsity:.2%})", flush=True)

        # Evaluate on ALL test datasets using the exact same loaders as training
        for test_dataset_name, (_, _, test_loader) in datasets_dict.items():
            metrics = trainer.test(fresh_model, test_loader)

            row = {
                "Train_Dataset":    train_dataset_name,
                "Test_Dataset":     test_dataset_name,
                "Pruning_Ratio":    ratio,
                "Actual_Sparsity":  round(actual_sparsity, 6),
                "GRU_Total_Params": gru_total,
                "GRU_Nonzero_Params": gru_nonzero,
                "SGCS":             round(metrics["sgcs"], 6),
                "Loss":             round(metrics["loss"], 6),
                "Recon_Loss":       round(metrics["recon_loss"], 6),
            }
            if "contrast_loss" in metrics:
                row["Contrast_Loss"] = round(metrics["contrast_loss"], 6)

            all_rows.append(row)
            print(f"  [{test_dataset_name}] SGCS: {metrics['sgcs']:.6f}  "
                  f"Loss: {metrics['loss']:.6f}", flush=True)

        # Clean up GPU memory between ratios
        del fresh_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save to CSV
    df = pd.DataFrame(all_rows)
    out_path = experiment_dir / "pruning_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPruning results saved to {out_path}", flush=True)

    # Also save a summary: ratio=0 vs best SGCS for quick sanity check
    if not df.empty:
        same_ds = df[df["Test_Dataset"] == train_dataset_name]
        if not same_ds.empty:
            baseline_sgcs = same_ds[same_ds["Pruning_Ratio"] == 0.0]["SGCS"].values
            best_sgcs     = same_ds["SGCS"].max()
            best_ratio    = same_ds.loc[same_ds["SGCS"].idxmax(), "Pruning_Ratio"]
            if len(baseline_sgcs) > 0:
                print(f"\nPruning summary [{train_dataset_name}]:", flush=True)
                print(f"  Baseline (ratio=0) SGCS : {baseline_sgcs[0]:.6f}", flush=True)
                print(f"  Best SGCS               : {best_sgcs:.6f} at ratio={best_ratio:.2f}", flush=True)
