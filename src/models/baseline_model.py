"""Baseline autoencoder model without CPC."""

import torch
import torch.nn as nn
from .blocks import QuantizedConv2d, QuantizedLinear, ResNetBlock, ResNetBlockDecoder


class BaselineEncoder(nn.Module):
    """Baseline encoder with ResNet blocks."""
    
    def __init__(self, compressed_size: int = 32, num_bits: int = 2):
        super(BaselineEncoder, self).__init__()
        self.compressed_size = compressed_size
        
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


class BaselineDecoder(nn.Module):
    """Baseline decoder with ResNet blocks."""
    
    def __init__(self, compressed_size: int = 32, num_bits: int = 2):
        super(BaselineDecoder, self).__init__()
        self.compressed_size = compressed_size
        
        # Initial flat layer
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
        x = self.conv2dLayer(x)
        
        return x


class BaselineAutoencoder(nn.Module):
    """Baseline autoencoder model."""
    
    def __init__(self, compressed_size: int = 32, num_bits: int = 2):
        super(BaselineAutoencoder, self).__init__()
        self.encoder = BaselineEncoder(compressed_size=compressed_size, num_bits=num_bits)
        self.decoder = BaselineDecoder(compressed_size=compressed_size, num_bits=num_bits)

    def forward(self, x):
        # Get input dimensions
        batch_size, channel_num, subband_num, port_num = x.shape
        
        # Encode
        latent = self.encoder(x)
        
        # Decode with the shape information
        output = self.decoder(latent, batch_size, subband_num, port_num, channel_num)
        
        return output
