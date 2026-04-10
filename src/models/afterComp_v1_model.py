"""CPC after compression V1 - applies predictive coding after reconstruction in decoder."""

import torch
import torch.nn as nn
from .blocks import QuantizedConv2d, QuantizedLinear, ResNetBlock, ResNetBlockDecoder
from .temporal import build_temporal_module


class AfterCompV1Encoder(nn.Module):
    """V1 Encoder - compress first, no CPC."""
    
    def __init__(self, compressed_size: int = 32, num_bits: int = 2):
        super(AfterCompV1Encoder, self).__init__()
        
        # Initial layers before ResNet blocks
        self.layersBeforeResNBEn = nn.Sequential(
            QuantizedConv2d(2, 64, kernel_size=3, padding=1, num_bits=num_bits),
            nn.ReLU()
        )
        
        # 6 ResNet blocks
        self.resBlock1 = ResNetBlock(64, num_bits)
        self.resBlock2 = ResNetBlock(64, num_bits)
        self.resBlock3 = ResNetBlock(64, num_bits)
        self.resBlock4 = ResNetBlock(64, num_bits)
        self.resBlock5 = ResNetBlock(64, num_bits)
        self.resBlock6 = ResNetBlock(64, num_bits)
        
        # Final layers
        self.layersAfterResNBEn = nn.Sequential(
            QuantizedConv2d(64, 32, kernel_size=1, num_bits=num_bits),  # 1x1 conv
            nn.Flatten(start_dim=1),
            QuantizedLinear(32 * 13 * 32, compressed_size, num_bits=num_bits)
        )

    def forward(self, x):
        # Initial conv block
        x = self.layersBeforeResNBEn(x)
        
        # ResNet blocks with skip connections
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.resBlock5(x)
        x = self.resBlock6(x)
        
        # Final layers
        x = self.layersAfterResNBEn(x)

        return x


class AfterCompV1Decoder(nn.Module):
    """V1 Decoder - reconstruct then apply CPC."""
    
    def __init__(self, compressed_size: int = 32, hidden_size: int = 64, num_bits: int = 2,
                 temporal_type: str = 'gru', num_layers: int = 1, dropout: float = 0.0,
                 num_heads: int = 4, feedforward_dim: int = 256, pred_steps: int = 5,
                 time_window: int = 10):
        super(AfterCompV1Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.pred_steps = pred_steps
        self.time_window = time_window
        
        # Reconstruction path
        self.flatLayer = QuantizedLinear(compressed_size, 64 * 13 * 32, num_bits=num_bits)
        
        # Initial convolutional block
        self.layersBeforeResNBDe = nn.Sequential(
            QuantizedConv2d(64, 64, kernel_size=3, padding=1, num_bits=num_bits),
            nn.ReLU()
        )
        
        # 6 ResNet blocks
        self.resBlock1 = ResNetBlockDecoder(64, num_bits)
        self.resBlock2 = ResNetBlockDecoder(64, num_bits)
        self.resBlock3 = ResNetBlockDecoder(64, num_bits)
        self.resBlock4 = ResNetBlockDecoder(64, num_bits)
        self.resBlock5 = ResNetBlockDecoder(64, num_bits)
        self.resBlock6 = ResNetBlockDecoder(64, num_bits)
        
        # Final 1x1 convolution to get output channels
        self.conv2dLayer = QuantizedConv2d(64, 2, kernel_size=1, num_bits=num_bits)

        # Predictive Coding - temporal module on reconstructed signal
        temporal_input_size = 2 * 13 * 32
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
        
    def forward(self, latent_message, batch_size, subband_num, port_num, channel_num):
        # Initial linear layer
        x = self.flatLayer(latent_message)
        
        # Reshape to match the expected dimensions
        x = x.view(batch_size, 64, subband_num, port_num)
        
        # Initial conv block
        x = self.layersBeforeResNBDe(x)
        
        # ResNet blocks with skip connections
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.resBlock5(x)
        x = self.resBlock6(x)
        
        # Final 1x1 convolution
        reconstructed = self.conv2dLayer(x)
        
        # Handle temporal windowing for CPC
        batch_size, c, h, w = reconstructed.shape
        
        if batch_size < self.time_window:
            return reconstructed, None, None
        
        num_complete_windows = batch_size // self.time_window
        complete_samples = num_complete_windows * self.time_window
        
        # Prepare data for temporal module
        x_temporal = reconstructed[:complete_samples].view(-1, self.time_window, c, h, w)
        x_temporal = x_temporal.view(-1, self.time_window, c * h * w)
        
        # Temporal processing
        last_context, _ = self.temporal_module(x_temporal)
        
        # Make predictions in hidden space
        predictions = []
        for pred_layer in self.pred_layers:
            pred = pred_layer(last_context)
            predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_steps, hidden_size)
        
        # Match original output format
        original_predictions = predictions.view(-1, self.hidden_size)
        
        return reconstructed, last_context, original_predictions


class AfterCompV1Autoencoder(nn.Module):
    """CPC after compression V1 autoencoder."""
    
    def __init__(self, compressed_size: int = 32, hidden_size: int = 64, num_bits: int = 2,
                 temporal_type: str = 'gru', num_layers: int = 1, dropout: float = 0.0,
                 num_heads: int = 4, feedforward_dim: int = 256, pred_steps: int = 5,
                 time_window: int = 10):
        super(AfterCompV1Autoencoder, self).__init__()
        
        self.encoder = AfterCompV1Encoder(
            compressed_size=compressed_size,
            num_bits=num_bits
        )
        
        self.decoder = AfterCompV1Decoder(
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
        
    def forward(self, x):
        # Encode 
        batch_size, channel_num, subband_num, port_num = x.shape
        compressed = self.encoder(x)
        
        # Check encoder output
        if compressed is None:
            return None, None, None
            
        # Decode and predict
        decoder_output = self.decoder(compressed, batch_size, subband_num, port_num, channel_num)
        
        # Check decoder output
        if decoder_output[1] is None or decoder_output[2] is None:
            return decoder_output[0], None, None
            
        reconstructed, context, predictions = decoder_output
        
        return reconstructed, context, predictions
