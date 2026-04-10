"""CPC before compression model - applies predictive coding before the compression bottleneck."""

import torch
import torch.nn as nn
from .blocks import QuantizedConv2d, QuantizedLinear, ResNetBlock, ResidualBlockDecoder
from .temporal import build_temporal_module


class BeforeCompEncoder(nn.Module):
    """Encoder with CPC before compression."""
    
    def __init__(self, compressed_size: int = 32, hidden_size: int = 64, num_bits: int = 2,
                 temporal_type: str = 'gru', num_layers: int = 1, dropout: float = 0.0,
                 num_heads: int = 4, feedforward_dim: int = 256, pred_steps: int = 5,
                 time_window: int = 10):
        super(BeforeCompEncoder, self).__init__()

        self.compressed_size = compressed_size
        self.hidden_size = hidden_size
        self.num_bits = num_bits
        self.pred_steps = pred_steps
        self.time_window = time_window
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            QuantizedConv2d(2, (hidden_size // 2), kernel_size=3, padding=1, num_bits=num_bits),
            nn.ReLU()
        )
        
        # ResNet blocks
        self.resBlock1 = ResNetBlock((hidden_size // 2), num_bits)
        self.resBlock2 = ResNetBlock((hidden_size // 2), num_bits)
        self.resBlock3 = ResNetBlock((hidden_size // 2), num_bits)
        
        # Last conv layer
        self.lastConv = QuantizedConv2d((hidden_size // 2), (hidden_size // 2), kernel_size=1, num_bits=num_bits)
        
        # Temporal module
        temporal_input_size = (hidden_size // 2) * 13 * 32
        self.temporal_module = build_temporal_module(
            module_type=temporal_type,
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_bits=num_bits
        )
        
        # Prediction layers
        self.pred_layers = nn.ModuleList([
            QuantizedLinear(hidden_size, hidden_size, num_bits=num_bits)
            for _ in range(pred_steps)
        ])
        
        # Compression layer
        self.compress = QuantizedLinear(hidden_size, compressed_size, num_bits=num_bits)

    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        
        # ResNet blocks
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        
        latest_x = self.lastConv(x)  # (batch, hidden_size//2, 13, 32)
        
        # Handle temporal windowing
        batch_size, c, h, w = latest_x.shape
        
        if batch_size < self.time_window:
            return None
        
        num_complete_windows = batch_size // self.time_window
        complete_samples = num_complete_windows * self.time_window
        x = latest_x[:complete_samples].view(-1, self.time_window, c, h, w)
        
        # Reshape for temporal module
        x_flat = x.reshape(-1, self.time_window, c * h * w)
        
        # Temporal processing
        last_context, _ = self.temporal_module(x_flat)  # (batch, hidden_size)
        
        # Make predictions
        predictions = []
        for pred_layer in self.pred_layers:
            pred = pred_layer(last_context)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_steps, hidden_size)
        
        original_context = last_context
        original_predictions = predictions.view(-1, self.hidden_size)
        
        # Compress predictions
        compressed_predictions = self.compress(original_predictions)  # (pred_steps*batch, compressed_size)
        
        return original_context, original_predictions, compressed_predictions


class BeforeCompDecoder(nn.Module):
    """Decoder for beforeComp model."""
    
    def __init__(self, compressed_size: int = 32, hidden_size: int = 64, num_bits: int = 2):
        super(BeforeCompDecoder, self).__init__()
        self.compressed_size = compressed_size
        self.hidden_size = hidden_size
        
        # Initial expansion from compressed to hidden size
        self.flatLayer = QuantizedLinear(compressed_size, hidden_size, num_bits=num_bits)
        
        # Initial block
        self.layersBeforeResBlock = nn.Sequential(
            QuantizedLinear(hidden_size, hidden_size, num_bits=num_bits),
            nn.ReLU()
        )
        
        # Residual blocks
        self.resBlock1 = ResidualBlockDecoder(hidden_size, num_bits)
        self.resBlock2 = ResidualBlockDecoder(hidden_size, num_bits)
        
        # Final layer
        self.layersAfterResBlock = nn.Sequential(
            QuantizedLinear(hidden_size, hidden_size, num_bits=num_bits),
        )
        
    def forward(self, x):
        # Initial linear expansion
        x = self.flatLayer(x)  # (batch, hidden_size)
        
        # Initial block
        x = self.layersBeforeResBlock(x)
        
        # Residual blocks
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        
        # Final layer
        x = self.layersAfterResBlock(x)
        
        return x


class BeforeCompAutoencoder(nn.Module):
    """CPC before compression autoencoder."""
    
    def __init__(self, compressed_size: int = 32, hidden_size: int = 64, num_bits: int = 2,
                 temporal_type: str = 'gru', num_layers: int = 1, dropout: float = 0.0,
                 num_heads: int = 4, feedforward_dim: int = 256, pred_steps: int = 5,
                 time_window: int = 10):
        super(BeforeCompAutoencoder, self).__init__()
        
        self.encoder = BeforeCompEncoder(
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
        
        self.decoder = BeforeCompDecoder(
            compressed_size=compressed_size,
            hidden_size=hidden_size,
            num_bits=num_bits
        )
        
    def forward(self, x):
        # Encode to get compressed representations
        compressed = self.encoder(x)

        # Check encoder output
        if compressed is None:
            return None, None, None
            
        original_context, original_predictions, compressed_predictions = compressed
        
        # Decode context and predictions
        decompressed_predictions = self.decoder(compressed_predictions)
        
        return original_context, original_predictions, decompressed_predictions
