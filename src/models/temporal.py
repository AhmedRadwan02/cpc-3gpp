"""Temporal modules for sequence modeling (GRU, Transformer, RNN, LSTM)."""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .blocks import QuantizedLinear, QuantizedGRU, QuantizedRNN, QuantizedLSTM


class TemporalModule(nn.Module):
    """Base class for temporal modules."""
    
    def __init__(self, input_size: int, hidden_size: int, num_bits: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_bits = num_bits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            context: Context vector of shape (batch, hidden_size)
            hidden_states: All hidden states of shape (batch, time, hidden_size)
        """
        raise NotImplementedError


class GRUModule(TemporalModule):
    """GRU-based temporal module."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 dropout: float = 0.0, num_bits: int = 2):
        super().__init__(input_size, hidden_size, num_bits)
        self.num_layers = num_layers
        self.gru = QuantizedGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_bits=num_bits,
            batch_first=True
        )
        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, features)
            
        Returns:
            context: (batch, hidden_size) - last hidden state
            hidden_states: (batch, time, hidden_size) - all hidden states
        """
        hidden_states, _ = self.gru(x)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        context = hidden_states[:, -1]  # Last timestep
        return context, hidden_states


class RNNModule(TemporalModule):
    """RNN-based temporal module (quantized, same structure as GRU for fair comparison)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, num_bits: int = 2):
        super().__init__(input_size, hidden_size, num_bits)
        self.num_layers = num_layers
        self.rnn = QuantizedRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_bits=num_bits,
            batch_first=True
        )
        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, features)

        Returns:
            context: (batch, hidden_size) - last hidden state
            hidden_states: (batch, time, hidden_size) - all hidden states
        """
        hidden_states, _ = self.rnn(x)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        context = hidden_states[:, -1]
        return context, hidden_states


class LSTMModule(TemporalModule):
    """LSTM-based temporal module (quantized, same structure as GRU for fair comparison)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, num_bits: int = 2):
        super().__init__(input_size, hidden_size, num_bits)
        self.num_layers = num_layers
        self.lstm = QuantizedLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_bits=num_bits,
            batch_first=True
        )
        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, features)

        Returns:
            context: (batch, hidden_size) - last hidden state
            hidden_states: (batch, time, hidden_size) - all hidden states
        """
        hidden_states, _ = self.lstm(x)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        context = hidden_states[:, -1]
        return context, hidden_states


class TransformerModule(TemporalModule):
    """Transformer-based temporal module."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 num_heads: int = 4, feedforward_dim: int = 256, dropout: float = 0.1,
                 num_bits: int = 2):
        super().__init__(input_size, hidden_size, num_bits)
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Project input to hidden_size if needed
        if input_size != hidden_size:
            self.input_proj = QuantizedLinear(input_size, hidden_size, num_bits=num_bits)
        else:
            self.input_proj = None
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = QuantizedLinear(hidden_size, hidden_size, num_bits=num_bits)
        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, features)

        Returns:
            context: (batch, hidden_size) - last hidden state (same as GRU/RNN/LSTM for fair comparison)
            hidden_states: (batch, time, hidden_size) - all hidden states
        """
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        hidden_states = self.transformer(x)
        
        # Project output
        hidden_states = self.output_proj(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        # Last timestep as context (same as GRU/RNN/LSTM for fair comparison)
        context = hidden_states[:, -1]
        
        return context, hidden_states


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def build_temporal_module(module_type: str, input_size: int, hidden_size: int,
                         num_layers: int = 1, dropout: float = 0.0,
                         num_heads: Optional[int] = None, 
                         feedforward_dim: Optional[int] = None,
                         num_bits: int = 2) -> TemporalModule:
    """Factory function to build temporal modules.
    
    Args:
        module_type: Type of temporal module ('gru', 'rnn', 'lstm', 'transformer')
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of layers
        dropout: Dropout rate
        num_heads: Number of attention heads (for Transformer)
        feedforward_dim: Feedforward dimension (for Transformer)
        num_bits: Number of bits for quantization
        
    Returns:
        TemporalModule instance
    """
    module_type = module_type.lower()
    
    if module_type == 'gru':
        return GRUModule(input_size, hidden_size, num_layers, dropout, num_bits)
    elif module_type == 'rnn':
        return RNNModule(input_size, hidden_size, num_layers, dropout, num_bits)
    elif module_type == 'lstm':
        return LSTMModule(input_size, hidden_size, num_layers, dropout, num_bits)
    elif module_type == 'transformer':
        if num_heads is None:
            num_heads = 4
        if feedforward_dim is None:
            feedforward_dim = hidden_size * 4
        return TransformerModule(input_size, hidden_size, num_layers, num_heads, 
                                feedforward_dim, dropout, num_bits)
    else:
        raise ValueError(f"Unknown temporal module type: {module_type}")
