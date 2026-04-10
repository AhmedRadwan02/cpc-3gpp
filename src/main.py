"""Main entry point for training CSI compression models."""

import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np
import pandas as pd
import sys

from .utils.config_utils import load_config, save_config, generate_experiment_name
from .utils.efficiency_metrics import compute_efficiency_metrics, save_efficiency_metrics
from .models import build_model
from .data_loader import load_all_datasets
from .trainer import Trainer
from .pruning import run_pruning_sweep


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(description='Train CSI compression models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--run', type=int, default=1, help='Run number')
    parser.add_argument('--train_dataset', type=str, default=None, 
                       help='Dataset to train on (NOKIA, OPPO, CAT, Mixed, or None for all)')
    parser.add_argument('--skip_pruning', action='store_true',
                       help='Skip post-training pruning sweep (beforeComp only)')
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Override dataset root (default: YAML data.path; else env DATA_ROOT)',
    )
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}", flush=True)
    config = load_config(args.config)
    data_override = args.data_path or os.environ.get('DATA_ROOT')
    if data_override:
        config.data.path = str(Path(data_override).expanduser().resolve())
        print(f"Dataset root (override): {config.data.path}", flush=True)
    
    # Set seed
    set_seed(config.experiment.seed + args.run)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
    
    # Load datasets
    print("\nLoading datasets...", flush=True)
    datasets_dict = load_all_datasets(
        data_path=config.data.path,
        datasets_config=[
            {'name': ds.name, 'path': ds.path, 'num_samples': ds.num_samples}
            for ds in config.data.datasets
        ],
        window_size=config.data.window_size,
        mixed_samples=config.data.mixed_samples_per_dataset,
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split
    )
    
    # Determine which datasets to train on
    if args.train_dataset:
        train_datasets = [args.train_dataset]
    else:
        train_datasets = list(datasets_dict.keys())
    
    # Store all results
    all_results = []

    # Config-level directory: one parent per (model + hyperparams + run) for traceability and ablations
    # e.g. experiments/beforeComp_gru_infonce_temp0.05_pred5_win10_2bit_run1/
    config_dir = Path(config.output.base_dir) / generate_experiment_name(config, args.run)
    config_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(config_dir / 'config.yaml'))
    # Record exact data path used so pruning can use the same path and match test_results.csv
    (config_dir / 'data_path_used.txt').write_text(str(Path(config.data.path).resolve()))
    print(f"Config directory: {config_dir}", flush=True)

    # Efficiency metrics once per config (same shape for all datasets; not dataset-specific)
    print("\nComputing efficiency metrics (params, inference time, FLOPs per sample)...", flush=True)
    try:
        model_for_eff = build_model(
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
            time_window=config.model.architecture.temporal_module.time_window
        )
        if torch.cuda.device_count() > 1:
            model_for_eff = nn.DataParallel(model_for_eff)
        model_for_eff = model_for_eff.to(device)
        sample_batch = next(iter(datasets_dict[train_datasets[0]][0]))[0]
        if sample_batch.dim() == 5 and config.model.type == "beforeComp":
            sample_batch = sample_batch.reshape(
                -1, sample_batch.size(2), sample_batch.size(3), sample_batch.size(4)
            )
        time_window = getattr(config.model.architecture.temporal_module, "time_window", 10)
        if config.model.type == "beforeComp" and sample_batch.size(0) < time_window:
            sample_batch = sample_batch.repeat(
                (time_window + sample_batch.size(0) - 1) // sample_batch.size(0), 1, 1, 1
            )[:time_window]
        eff_metrics = compute_efficiency_metrics(
            model_for_eff, config.model.type, sample_batch, device
        )
        eff_path = save_efficiency_metrics(eff_metrics, config_dir)
        print(f"  Encoder params: {eff_metrics['encoder_params']:,}", flush=True)
        print(f"  Decoder params: {eff_metrics['decoder_params']:,}", flush=True)
        print(f"  Encoder time (ms/sample): {eff_metrics['encoder_time_ms_per_sample']}", flush=True)
        print(f"  Decoder time (ms/sample): {eff_metrics['decoder_time_ms_per_sample']}", flush=True)
        if eff_metrics.get("encoder_flops_per_sample") is not None:
            print(f"  Encoder FLOPs/sample: {eff_metrics['encoder_flops_per_sample']:,}", flush=True)
        if eff_metrics.get("decoder_flops_per_sample") is not None:
            print(f"  Decoder FLOPs/sample: {eff_metrics['decoder_flops_per_sample']:,}", flush=True)
        print(f"  Saved to {eff_path} (shared for all datasets, same input shape)", flush=True)
    except Exception as e:
        print(f"  Efficiency metrics failed: {e}", flush=True)

    # Train on each dataset (each gets a subfolder: NOKIA, OPPO, CAT, Mixed)
    for train_dataset_name in train_datasets:
        print(f"\n{'='*70}", flush=True)
        print(f"Run {args.run}: Training on {train_dataset_name} dataset", flush=True)
        print(f"{'='*70}", flush=True)

        # Dataset-level directory inside config dir
        experiment_dir = config_dir / train_dataset_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment directory: {experiment_dir}", flush=True)
        
        # Build model
        print("\nBuilding model...", flush=True)
        model = build_model(
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
            time_window=config.model.architecture.temporal_module.time_window
        )
        
        # Print model info
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        print(f"Encoder parameters: {encoder_params:,}", flush=True)
        print(f"Decoder parameters: {decoder_params:,}", flush=True)
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
        
        # Move to device and wrap with DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        
        # Get train and validation loaders
        train_loader, val_loader, _ = datasets_dict[train_dataset_name]
        
        # Create trainer
        trainer = Trainer(config, experiment_dir, device)
        
        # Train model
        print("\nStarting training...", flush=True)
        model = trainer.train(model, train_loader, val_loader)
        
        # Test on all datasets
        print("\nTesting on all datasets...", flush=True)
        for test_dataset_name, (_, _, test_loader) in datasets_dict.items():
            print(f"\nEvaluating on {test_dataset_name} dataset...", flush=True)
            test_metrics = trainer.test(model, test_loader)
            
            result_row = {
                'Run': args.run,
                'Train_Dataset': train_dataset_name,
                'Test_Dataset': test_dataset_name,
                'hidden_size': config.model.architecture.hidden_size,
                'SGCS': round(test_metrics['sgcs'], 6),
                'Loss': round(test_metrics['loss'], 6),
                'Recon_Loss': round(test_metrics['recon_loss'], 6)
            }
            
            if 'contrast_loss' in test_metrics:
                result_row['Contrast_Loss'] = round(test_metrics['contrast_loss'], 6)
            
            all_results.append(result_row)
            
            print(f"Results for {test_dataset_name}:", flush=True)
            print(f"  SGCS: {test_metrics['sgcs']:.6f}", flush=True)
            print(f"  Loss: {test_metrics['loss']:.6f}", flush=True)
        
        # Save results for this training dataset
        results_df = pd.DataFrame([r for r in all_results if r['Train_Dataset'] == train_dataset_name])
        results_path = experiment_dir / 'test_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}", flush=True)

        # ------------------------------------------------------------------
        # Post-training pruning sweep (beforeComp only, unless --skip_pruning)
        # Runs immediately after training using the same test_loaders already
        # in memory — this guarantees ratio=0 SGCS matches test_results.csv.
        # ------------------------------------------------------------------
        if config.model.type == "beforeComp" and not args.skip_pruning:
            print(f"\nRunning post-training pruning sweep for {train_dataset_name}...", flush=True)
            try:
                run_pruning_sweep(
                    model=model,
                    trainer=trainer,
                    datasets_dict=datasets_dict,
                    train_dataset_name=train_dataset_name,
                    experiment_dir=experiment_dir,
                    config=config,
                    device=device,
                )
            except Exception as e:
                print(f"  Pruning sweep failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

    # Save all results at config level (one all_results.csv per config, inside the config dir)
    if len(all_results) > 0:
        all_results_df = pd.DataFrame(all_results)
        all_results_path = config_dir / 'all_results.csv'
        all_results_df.to_csv(all_results_path, index=False)
        print(f"\nAll results saved to {all_results_path}", flush=True)
    
    print("\nTraining completed!", flush=True)


if __name__ == '__main__':
    main()