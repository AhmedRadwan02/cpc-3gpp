"""Model factory for building different autoencoder variants."""

from .baseline_model import BaselineAutoencoder
from .beforeComp_model import BeforeCompAutoencoder
from .afterComp_v1_model import AfterCompV1Autoencoder
from .afterComp_v2_model import AfterCompV2Autoencoder


def build_model(model_type: str, compressed_size: int = 32, hidden_size: int = 64,
                num_bits: int = 2, temporal_type: str = 'gru', num_layers: int = 1,
                dropout: float = 0.0, num_heads: int = 4, feedforward_dim: int = 256,
                pred_steps: int = 5, time_window: int = 10):
    """Factory function to build models based on type.
    
    Args:
        model_type: Type of model ('baseline', 'beforeComp', 'afterComp_v1', 'afterComp_v2')
        compressed_size: Size of compressed latent representation
        hidden_size: Hidden size for temporal modules
        num_bits: Number of bits for quantization
        temporal_type: Type of temporal module ('gru', 'rnn', 'lstm', 'transformer')
        num_layers: Number of layers in temporal module
        dropout: Dropout rate
        num_heads: Number of attention heads (for Transformer)
        feedforward_dim: Feedforward dimension (for Transformer)
        pred_steps: Number of prediction steps for CPC
        time_window: Window size for internal temporal grouping (CPC models)
        
    Returns:
        Model instance
    """
    if model_type == 'baseline':
        return BaselineAutoencoder(
            compressed_size=compressed_size,
            num_bits=num_bits
        )
    elif model_type == 'beforeComp':
        return BeforeCompAutoencoder(
            compressed_size=compressed_size,
            hidden_size=hidden_size,
            num_bits=num_bits,
            temporal_type=temporal_type,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            pred_steps=pred_steps,
            time_window=time_window
        )
    elif model_type == 'afterComp_v1':
        return AfterCompV1Autoencoder(
            compressed_size=compressed_size,
            hidden_size=hidden_size,
            num_bits=num_bits,
            temporal_type=temporal_type,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            pred_steps=pred_steps,
            time_window=time_window
        )
    elif model_type == 'afterComp_v2':
        return AfterCompV2Autoencoder(
            compressed_size=compressed_size,
            hidden_size=hidden_size,
            num_bits=num_bits,
            temporal_type=temporal_type,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            pred_steps=pred_steps,
            time_window=time_window
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    'BaselineAutoencoder',
    'BeforeCompAutoencoder',
    'AfterCompV1Autoencoder',
    'AfterCompV2Autoencoder',
    'build_model'
]
