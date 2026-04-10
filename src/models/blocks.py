"""Shared neural network blocks for all models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedFunction(torch.autograd.Function):
    """Custom autograd function for weight quantization with straight-through estimator."""
    
    @staticmethod
    def forward(ctx, input, num_bits=2):
        ctx.num_bits = num_bits
        # Calculate step size for quantization
        min_val = input.min()
        max_val = input.max()
        step_size = (max_val - min_val) / (2**num_bits - 1)
        
        # Quantize the input
        quantized = torch.round((input - min_val) / step_size) * step_size + min_val
        
        # Save for backward pass
        ctx.save_for_backward(input, quantized)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients through unchanged
        input, quantized = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None


class QuantizedConv2d(nn.Conv2d):
    """2D convolution with quantized weights."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, num_bits=2):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.num_bits = num_bits

    def forward(self, x):
        quantized_weight = QuantizedFunction.apply(self.weight, self.num_bits)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class QuantizedLinear(nn.Linear):
    """Linear layer with quantized weights."""
    
    def __init__(self, in_features, out_features, num_bits=2):
        super().__init__(in_features, out_features)
        self.num_bits = num_bits

    def forward(self, x):
        quantized_weight = QuantizedFunction.apply(self.weight, self.num_bits)
        return F.linear(x, quantized_weight, self.bias)


class QuantizedGRU(nn.GRU):
    """GRU with quantized weights."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, num_bits=2, bias=True, batch_first=True):
        super().__init__(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
        self.num_bits = num_bits

    def forward(self, x, hx=None):
        # Create quantized copies of weights without modifying the original parameters
        quantized_weights = {}
        for layer in range(self.num_layers):
            ih_weight = getattr(self, f'weight_ih_l{layer}')
            hh_weight = getattr(self, f'weight_hh_l{layer}')
            
            # Store original weights
            original_ih = ih_weight.data.clone()
            original_hh = hh_weight.data.clone()
            
            # Quantize weights
            quantized_ih = QuantizedFunction.apply(ih_weight, self.num_bits)
            quantized_hh = QuantizedFunction.apply(hh_weight, self.num_bits)
            
            # Temporarily replace weights
            ih_weight.data = quantized_ih
            hh_weight.data = quantized_hh
            
            # Store for restoration
            quantized_weights[f'ih_{layer}'] = (ih_weight, original_ih)
            quantized_weights[f'hh_{layer}'] = (hh_weight, original_hh)
        
        try:
            # Perform forward pass with quantized weights
            output = super().forward(x, hx)
        finally:
            # Restore original weights
            for layer in range(self.num_layers):
                ih_weight, original_ih = quantized_weights[f'ih_{layer}']
                hh_weight, original_hh = quantized_weights[f'hh_{layer}']
                ih_weight.data = original_ih
                hh_weight.data = original_hh
        
        return output


class QuantizedRNN(nn.RNN):
    """RNN with quantized weights (same structure as QuantizedGRU for fair comparison)."""

    def __init__(self, input_size, hidden_size, num_layers=1, num_bits=2, bias=True, batch_first=True):
        super().__init__(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
        self.num_bits = num_bits

    def forward(self, x, hx=None):
        saved = []
        for layer in range(self.num_layers):
            ih_weight = getattr(self, f'weight_ih_l{layer}')
            hh_weight = getattr(self, f'weight_hh_l{layer}')
            saved.append((ih_weight.data.clone(), hh_weight.data.clone()))
            ih_weight.data = QuantizedFunction.apply(ih_weight, self.num_bits)
            hh_weight.data = QuantizedFunction.apply(hh_weight, self.num_bits)
        try:
            output = super().forward(x, hx)
        finally:
            for layer in range(self.num_layers):
                ih_weight = getattr(self, f'weight_ih_l{layer}')
                hh_weight = getattr(self, f'weight_hh_l{layer}')
                ih_weight.data, hh_weight.data = saved[layer]
        return output


class QuantizedLSTM(nn.LSTM):
    """LSTM with quantized weights (same structure as QuantizedGRU for fair comparison)."""

    def __init__(self, input_size, hidden_size, num_layers=1, num_bits=2, bias=True, batch_first=True):
        super().__init__(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first)
        self.num_bits = num_bits

    def forward(self, x, hx=None):
        saved = []
        for layer in range(self.num_layers):
            ih_weight = getattr(self, f'weight_ih_l{layer}')
            hh_weight = getattr(self, f'weight_hh_l{layer}')
            saved.append((ih_weight.data.clone(), hh_weight.data.clone()))
            ih_weight.data = QuantizedFunction.apply(ih_weight, self.num_bits)
            hh_weight.data = QuantizedFunction.apply(hh_weight, self.num_bits)
        try:
            output = super().forward(x, hx)
        finally:
            for layer in range(self.num_layers):
                ih_weight = getattr(self, f'weight_ih_l{layer}')
                hh_weight = getattr(self, f'weight_hh_l{layer}')
                ih_weight.data, hh_weight.data = saved[layer]
        return output


class ResNetBlock(nn.Module):
    """ResNet block with skip connection for encoder."""
    
    def __init__(self, channels, num_bits):
        super(ResNetBlock, self).__init__()
        self.conv1 = QuantizedConv2d(channels, channels, kernel_size=3, padding=1, num_bits=num_bits)
        self.relu = nn.ReLU()
        self.conv2 = QuantizedConv2d(channels, channels, kernel_size=3, padding=1, num_bits=num_bits)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity  # Skip connection
        out = self.relu(out)
        return out


class ResNetBlockDecoder(nn.Module):
    """ResNet block with skip connection for decoder."""
    
    def __init__(self, channels, num_bits):
        super(ResNetBlockDecoder, self).__init__()
        self.conv1 = QuantizedConv2d(channels, channels, kernel_size=3, padding=1, num_bits=num_bits)
        self.relu = nn.ReLU()
        self.conv2 = QuantizedConv2d(channels, channels, kernel_size=3, padding=1, num_bits=num_bits)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity  # Skip connection
        out = self.relu(out)
        return out


class ResidualBlockDecoder(nn.Module):
    """Residual block with linear layers for decoder (used in beforeComp model)."""
    
    def __init__(self, hidden_size, num_bits):
        super(ResidualBlockDecoder, self).__init__()
        self.linear1 = QuantizedLinear(hidden_size, hidden_size, num_bits=num_bits)
        self.relu = nn.ReLU()
        self.linear2 = QuantizedLinear(hidden_size, hidden_size, num_bits=num_bits)
        
    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = out + identity  # Skip connection
        out = self.relu(out)
        return out
