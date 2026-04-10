"""Configuration utilities for loading and validating experiment configs."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str
    description: str = ""
    seed: int = 42
    num_runs: int = 5


@dataclass
class TemporalModuleConfig:
    """Temporal module configuration."""
    type: str = "gru"  # gru, transformer, rnn, lstm
    num_layers: int = 1
    dropout: float = 0.0
    time_window: int = 10  # Window size for internal temporal grouping
    # Transformer-specific
    num_heads: Optional[int] = None
    feedforward_dim: Optional[int] = None


@dataclass
class PredictionConfig:
    """Prediction configuration."""
    num_steps: int = 5
    use_residual: bool = False


@dataclass
class ArchitectureConfig:
    """Model architecture configuration."""
    compressed_size: int = 32
    hidden_size: int = 64
    num_bits: int = 2
    temporal_module: TemporalModuleConfig = field(default_factory=TemporalModuleConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str  # baseline, beforeComp, afterComp_v1, afterComp_v2
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)


@dataclass
class LossConfig:
    """Loss function configuration."""
    reconstruction_type: str = "sgcs"  # sgcs, mse, l1
    reconstruction_weight: float = 0.5
    contrastive_type: str = "infonce"  # infonce, simclr, vicreg, none
    contrastive_weight: float = 0.5
    temperature: float = 0.05
    # VICReg-specific
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "none"  # none, cosine, step
    step_size: Optional[int] = None
    gamma: Optional[float] = None


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: int = 25
    metric: str = "val_sgcs"
    mode: str = "max"  # max or min


@dataclass
class TrainingConfig:
    """Training configuration."""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    epochs: int = 150
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    batch_size: int = 256


@dataclass
class DatasetInfo:
    """Individual dataset information."""
    name: str
    path: str
    num_samples: int


@dataclass
class DataConfig:
    """Data configuration."""
    path: str
    datasets: list[DatasetInfo] = field(default_factory=list)
    mixed_samples_per_dataset: int = 100000
    window_size: Optional[int] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "./experiments"
    save_best_only: bool = True
    save_frequency: int = 10


@dataclass
class Config:
    """Main configuration class."""
    experiment: ExperimentConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with all settings
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse nested configurations
    experiment = ExperimentConfig(**config_dict.get('experiment', {}))
    
    # Model config
    model_dict = config_dict.get('model', {})
    arch_dict = model_dict.get('architecture', {})
    temporal_dict = arch_dict.get('temporal_module', {})
    prediction_dict = arch_dict.get('prediction', {})
    
    temporal_module = TemporalModuleConfig(
        type=temporal_dict.get('type', 'gru'),
        num_layers=temporal_dict.get('num_layers', 1),
        dropout=temporal_dict.get('dropout', 0.0),
        time_window=temporal_dict.get('time_window', 10),
        num_heads=temporal_dict.get('num_heads'),
        feedforward_dim=temporal_dict.get('feedforward_dim')
    )
    prediction = PredictionConfig(**prediction_dict)
    architecture = ArchitectureConfig(
        compressed_size=arch_dict.get('compressed_size', 32),
        hidden_size=arch_dict.get('hidden_size', 64),
        num_bits=arch_dict.get('num_bits', 2),
        temporal_module=temporal_module,
        prediction=prediction
    )
    model = ModelConfig(
        type=model_dict.get('type', 'baseline'),
        architecture=architecture
    )
    
    # Loss config
    loss_dict = config_dict.get('loss', {})
    recon_dict = loss_dict.get('reconstruction', {})
    contrast_dict = loss_dict.get('contrastive', {})
    loss = LossConfig(
        reconstruction_type=recon_dict.get('type', 'sgcs'),
        reconstruction_weight=recon_dict.get('weight', 0.5),
        contrastive_type=contrast_dict.get('type', 'infonce'),
        contrastive_weight=contrast_dict.get('weight', 0.5),
        temperature=contrast_dict.get('temperature', 0.05),
        sim_coeff=contrast_dict.get('sim_coeff', 25.0),
        std_coeff=contrast_dict.get('std_coeff', 25.0),
        cov_coeff=contrast_dict.get('cov_coeff', 1.0)
    )
    
    # Training config
    train_dict = config_dict.get('training', {})
    opt_dict = train_dict.get('optimizer', {})
    sched_dict = train_dict.get('scheduler', {})
    es_dict = train_dict.get('early_stopping', {})
    
    optimizer = OptimizerConfig(**opt_dict)
    scheduler = SchedulerConfig(**sched_dict)
    early_stopping = EarlyStoppingConfig(**es_dict)
    training = TrainingConfig(
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=train_dict.get('epochs', 150),
        early_stopping=early_stopping,
        batch_size=train_dict.get('batch_size', 256)
    )
    
    # Data config
    data_dict = config_dict.get('data', {})
    datasets = [DatasetInfo(**ds) for ds in data_dict.get('datasets', [])]
    splits = data_dict.get('splits', {})
    data = DataConfig(
        path=data_dict.get('path', ''),
        datasets=datasets,
        mixed_samples_per_dataset=data_dict.get('mixed_samples_per_dataset', 100000),
        window_size=data_dict.get('window_size'),
        train_split=splits.get('train', 0.8),
        val_split=splits.get('val', 0.1),
        test_split=splits.get('test', 0.1)
    )
    
    # Output config
    output_dict = config_dict.get('output', {})
    output = OutputConfig(**output_dict)
    
    return Config(
        experiment=experiment,
        model=model,
        loss=loss,
        training=training,
        data=data,
        output=output
    )


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dict (manual serialization for dataclasses)
    config_dict = {
        'experiment': {
            'name': config.experiment.name,
            'description': config.experiment.description,
            'seed': config.experiment.seed,
            'num_runs': config.experiment.num_runs
        },
        'model': {
            'type': config.model.type,
            'architecture': {
                'compressed_size': config.model.architecture.compressed_size,
                'hidden_size': config.model.architecture.hidden_size,
                'num_bits': config.model.architecture.num_bits,
                'temporal_module': {
                    'type': config.model.architecture.temporal_module.type,
                    'num_layers': config.model.architecture.temporal_module.num_layers,
                    'dropout': config.model.architecture.temporal_module.dropout,
                    'time_window': config.model.architecture.temporal_module.time_window,
                    'num_heads': config.model.architecture.temporal_module.num_heads,
                    'feedforward_dim': config.model.architecture.temporal_module.feedforward_dim
                },
                'prediction': {
                    'num_steps': config.model.architecture.prediction.num_steps,
                    'use_residual': config.model.architecture.prediction.use_residual
                }
            }
        },
        'loss': {
            'reconstruction': {
                'type': config.loss.reconstruction_type,
                'weight': config.loss.reconstruction_weight
            },
            'contrastive': {
                'type': config.loss.contrastive_type,
                'weight': config.loss.contrastive_weight,
                'temperature': config.loss.temperature,
                'sim_coeff': config.loss.sim_coeff,
                'std_coeff': config.loss.std_coeff,
                'cov_coeff': config.loss.cov_coeff
            }
        },
        'training': {
            'optimizer': {
                'type': config.training.optimizer.type,
                'lr': config.training.optimizer.lr,
                'weight_decay': config.training.optimizer.weight_decay
            },
            'scheduler': {
                'type': config.training.scheduler.type,
                'step_size': config.training.scheduler.step_size,
                'gamma': config.training.scheduler.gamma
            },
            'epochs': config.training.epochs,
            'early_stopping': {
                'patience': config.training.early_stopping.patience,
                'metric': config.training.early_stopping.metric,
                'mode': config.training.early_stopping.mode
            },
            'batch_size': config.training.batch_size
        },
        'data': {
            'path': config.data.path,
            'datasets': [
                {'name': ds.name, 'path': ds.path, 'num_samples': ds.num_samples}
                for ds in config.data.datasets
            ],
            'mixed_samples_per_dataset': config.data.mixed_samples_per_dataset,
            'window_size': config.data.window_size,
            'splits': {
                'train': config.data.train_split,
                'val': config.data.val_split,
                'test': config.data.test_split
            }
        },
        'output': {
            'base_dir': config.output.base_dir,
            'save_best_only': config.output.save_best_only,
            'save_frequency': config.output.save_frequency
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def generate_experiment_name(config: Config, run_num: int) -> str:
    """Generate descriptive experiment folder name from config (config-level only, no dataset).
    Used as parent dir for traceability and ablations (temp, pred, loss, time_window, etc.).

    Args:
        config: Configuration object
        run_num: Run number

    Returns:
        Experiment folder name, e.g. beforeComp_gru_sgcs_infonce_temp0.05_pred5_win10_bottleneck32_hidden32_2bit_run1
    """
    parts = [config.model.type]

    # Add temporal module if not baseline
    if config.model.type != "baseline":
        parts.append(config.model.architecture.temporal_module.type)

    # Reconstruction loss (for ablations: sgcs vs mse vs l1)
    parts.append(config.loss.reconstruction_type)

    # Contrastive loss and rest only for non-baseline
    if config.model.type != "baseline":
        # Contrastive loss (for ablations: infonce vs simclr vs vicreg, temperature)
        if config.loss.contrastive_type != "none":
            parts.append(config.loss.contrastive_type)
            if config.loss.contrastive_type in ("infonce", "simclr"):
                parts.append(f"temp{config.loss.temperature}")

        # Prediction steps (for ablations)
        parts.append(f"pred{config.model.architecture.prediction.num_steps}")

        # Time window (for ablations)
        parts.append(f"win{config.model.architecture.temporal_module.time_window}")

    # Bottleneck (compressed_size) for ablations and to avoid folder collision
    parts.append(f"bottleneck{config.model.architecture.compressed_size}")

    # Hidden size (for tuning temporal module / architecture)
    parts.append(f"hidden{config.model.architecture.hidden_size}")

    # Bits (always included for all models)
    parts.append(f"{config.model.architecture.num_bits}bit")

    # Run number
    parts.append(f"run{run_num}")

    return "_".join(parts)
