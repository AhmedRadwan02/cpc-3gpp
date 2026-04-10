"""Efficiency metrics: encoder/decoder params, per-sample inference time, and FLOPs."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _get_actual_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in module.parameters())


def _encoder_time_per_sample(
    encoder: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_repeat: int = 100,
) -> float:
    """Encoder inference time per sample (ms). Uses first dimension as batch."""
    encoder.eval()
    x = x.to(device)
    batch_size = x.size(0)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = encoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            _ = encoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / num_repeat / batch_size) * 1000.0  # ms per sample


def _decoder_time_per_sample(
    decoder: nn.Module,
    decoder_inputs: Tuple[Any, ...],
    device: torch.device,
    batch_size: int,
    num_warmup: int = 10,
    num_repeat: int = 100,
) -> float:
    """Decoder inference time per sample (ms)."""
    decoder.eval()
    decoder_inputs = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in decoder_inputs
    )
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = decoder(*decoder_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            _ = decoder(*decoder_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / num_repeat / batch_size) * 1000.0  # ms per sample


def _flops_conv2d(module: nn.Module, _inp: Any, out: torch.Tensor) -> int:
    """FLOPs for Conv2d or QuantizedConv2d: 2 * C_in * C_out * K*K * H_out * W_out."""
    if out.dim() != 4:
        return 0
    _, c_out, h_out, w_out = out.shape
    in_c = getattr(module, "in_channels", 0)
    out_c = getattr(module, "out_channels", c_out)
    k = getattr(module, "kernel_size", (1, 1))
    if isinstance(k, int):
        k = (k, k)
    k1, k2 = k[0], k[1]
    return 2 * in_c * out_c * k1 * k2 * h_out * w_out


def _flops_linear(module: nn.Module, _inp: Any, out: torch.Tensor) -> int:
    """FLOPs for Linear or QuantizedLinear: 2 * in_features * out_features * batch."""
    if out.dim() != 2:
        return 0
    batch = out.shape[0]
    in_f = getattr(module, "in_features", 0)
    out_f = getattr(module, "out_features", out.shape[1])
    return 2 * in_f * out_f * batch


def _flops_gru(module: nn.Module, inp: Any, out: torch.Tensor) -> int:
    """FLOPs for GRU/QuantizedGRU: 6 * (input_size*hidden + hidden*hidden) per step, * batch * time."""
    if isinstance(inp, (tuple, list)):
        x = inp[0]
    else:
        x = inp
    if not isinstance(x, torch.Tensor) or x.dim() < 2:
        return 0
    batch, *rest = x.shape
    seq_len = rest[0] if len(rest) >= 1 else 1
    input_size = getattr(module, "input_size", x.shape[-1])
    hidden_size = getattr(module, "hidden_size", out.shape[-1])
    # 3 gates, each: input projection (input_size * hidden) + hidden projection (hidden * hidden), mult-add
    per_step = 3 * 2 * (input_size * hidden_size + hidden_size * hidden_size)
    return per_step * batch * seq_len


def _flops_rnn_lstm(module: nn.Module, inp: Any, out: torch.Tensor) -> int:
    """FLOPs for RNN or LSTM (one layer): similar to GRU, RNN=1 gate, LSTM=4 gates."""
    if isinstance(inp, (tuple, list)):
        x = inp[0]
    else:
        x = inp
    if not isinstance(x, torch.Tensor) or x.dim() < 2:
        return 0
    batch, *rest = x.shape
    seq_len = rest[0] if len(rest) >= 1 else 1
    input_size = x.shape[-1]
    hidden_size = getattr(module, "hidden_size", out.shape[-1])
    num_layers = getattr(module, "num_layers", 1)
    # RNN: 2*(input*hidden + hidden*hidden) per step; LSTM: 4*2*(...); treat as 2 gates for RNN
    gates = 4 if "LSTM" in type(module).__name__ else 1
    per_step = gates * 2 * (input_size * hidden_size + hidden_size * hidden_size)
    return per_step * batch * seq_len * num_layers


def _flops_via_hooks(module: nn.Module, run_fn: Callable[..., Any], run_args: Tuple[Any, ...]) -> int:
    """
    Run one forward and sum FLOPs from hooks on known layer types.
    Works with QuantizedConv2d, QuantizedLinear, QuantizedGRU, and standard nn layers.
    """
    total_flops = [0]  # use list so hook can mutate

    def _hook(m: nn.Module, inp: Any, out: Any) -> None:
        if isinstance(out, (tuple, list)):
            out = out[0] if out else None
        if not isinstance(out, torch.Tensor):
            return
        cls_name = type(m).__name__
        if "Conv2d" in cls_name or isinstance(m, nn.Conv2d):
            total_flops[0] += _flops_conv2d(m, inp, out)
        elif "Linear" in cls_name or isinstance(m, nn.Linear):
            total_flops[0] += _flops_linear(m, inp, out)
        elif "GRU" in cls_name or isinstance(m, nn.GRU):
            total_flops[0] += _flops_gru(m, inp, out)
        elif "RNN" in cls_name and "LSTM" not in cls_name:
            total_flops[0] += _flops_rnn_lstm(m, inp, out)
        elif "LSTM" in cls_name:
            total_flops[0] += _flops_rnn_lstm(m, inp, out)

    hooks = []
    for name, child in module.named_modules():
        if name == "":
            continue
        hooks.append(child.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            run_fn(*run_args)
    finally:
        for h in hooks:
            h.remove()

    return total_flops[0]


def _flops_encoder(encoder: nn.Module, x: torch.Tensor) -> Optional[int]:
    """FLOPs for encoder forward (total for the given input)."""
    try:
        flops = _flops_via_hooks(encoder, encoder.forward, (x,))
        return flops if flops > 0 else None
    except Exception:
        return None


def _flops_decoder(decoder: nn.Module, decoder_inputs: Tuple[Any, ...], batch_size: int) -> Optional[int]:
    """FLOPs for decoder forward (total); per-sample = total / batch_size."""
    try:
        flops = _flops_via_hooks(decoder, decoder.forward, decoder_inputs)
        return flops if flops > 0 else None
    except Exception:
        return None


def _prepare_decoder_inputs(
    model_type: str,
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> Tuple[Tuple[Any, ...], int]:
    """
    Get decoder inputs and batch size for one forward.
    Returns (decoder_inputs_tuple, batch_size_for_per_sample).
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        if model_type == "baseline":
            # decoder(latent, batch_size, subband_num, port_num, channel_num)
            latent = model.encoder(x)
            batch_size, channel_num, subband_num, port_num = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            return (latent, batch_size, subband_num, port_num, channel_num), batch_size

        if model_type == "beforeComp":
            # decoder(compressed_predictions); encoder returns (ctx, preds, compressed)
            out = model.encoder(x)
            if out is None:
                raise ValueError("beforeComp encoder returned None (batch too small for time_window)")
            _, _, compressed_predictions = out
            batch_size = compressed_predictions.size(0)  # (batch * pred_steps) or similar
            # Per-sample for beforeComp: one "sample" = one temporal window
            num_windows = x.size(0) // model.encoder.time_window
            return (compressed_predictions,), max(1, num_windows)

        if model_type in ("afterComp_v1", "afterComp_v2"):
            # decoder(latent_message, batch_size, subband_num, port_num, channel_num)
            latent = model.encoder(x)
            batch_size, channel_num, subband_num, port_num = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            return (latent, batch_size, subband_num, port_num, channel_num), batch_size

    raise ValueError(f"Unknown model type: {model_type}")


def compute_efficiency_metrics(
    model: nn.Module,
    model_type: str,
    sample_batch: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_repeat: int = 100,
) -> Dict[str, Any]:
    """
    Compute encoder/decoder parameter counts, per-sample inference time (ms), and FLOPs per sample.

    Args:
        model: Full model (encoder + decoder).
        model_type: One of 'baseline', 'beforeComp', 'afterComp_v1', 'afterComp_v2'.
        sample_batch: One batch from the dataloader (e.g. shape (B, 2, 13, 32) or (B, T, 2, 13, 32)).
        device: Device to run on.
        num_warmup: Warmup iterations for timing.
        num_repeat: Repeat count for timing.

    Returns:
        Dict with encoder_params, decoder_params, encoder_time_ms_per_sample,
        decoder_time_ms_per_sample, encoder_flops_per_sample, decoder_flops_per_sample
        (flops may be None if thop is unavailable or fails).
    """
    actual = _get_actual_model(model)
    encoder = actual.encoder
    decoder = actual.decoder

    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)

    # Flatten time into batch if 5D (B, T, C, H, W) -> (B*T, C, H, W)
    if sample_batch.dim() == 5:
        sample_batch = sample_batch.reshape(
            -1, sample_batch.size(2), sample_batch.size(3), sample_batch.size(4)
        )

    # Ensure we have a batch on device for beforeComp (need enough frames for time_window)
    if model_type == "beforeComp" and sample_batch.dim() == 4:
        time_window = getattr(actual.encoder, "time_window", 10)
        if sample_batch.size(0) < time_window:
            # Repeat to get at least time_window frames
            n = (time_window + sample_batch.size(0) - 1) // sample_batch.size(0)
            sample_batch = sample_batch.repeat(n, 1, 1, 1)[:time_window]
    x = sample_batch.to(device)

    # Encoder time (per sample)
    encoder_time_ms = _encoder_time_per_sample(encoder, x, device, num_warmup, num_repeat)

    # Decoder inputs and effective batch size for per-sample
    try:
        decoder_inputs, effective_batch = _prepare_decoder_inputs(model_type, actual, x, device)
    except Exception as e:
        return {
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "encoder_time_ms_per_sample": round(encoder_time_ms, 4),
            "decoder_time_ms_per_sample": None,
            "encoder_flops_per_sample": None,
            "decoder_flops_per_sample": None,
            "error": str(e),
        }

    # Decoder time (per sample)
    decoder_time_ms = _decoder_time_per_sample(
        decoder, decoder_inputs, device, effective_batch, num_warmup, num_repeat
    )

    # FLOPs (per sample)
    encoder_flops_total = _flops_encoder(encoder, x)
    decoder_flops_total = _flops_decoder(decoder, decoder_inputs, effective_batch)

    encoder_flops_per_sample = int(encoder_flops_total / x.size(0)) if encoder_flops_total is not None else None
    decoder_flops_per_sample = (
        int(decoder_flops_total / effective_batch) if decoder_flops_total is not None else None
    )

    return {
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "encoder_time_ms_per_sample": round(encoder_time_ms, 4),
        "decoder_time_ms_per_sample": round(decoder_time_ms, 4),
        "encoder_flops_per_sample": encoder_flops_per_sample,
        "decoder_flops_per_sample": decoder_flops_per_sample,
    }


def save_efficiency_metrics(metrics: Dict[str, Any], experiment_dir: Path) -> Path:
    """Save efficiency metrics to experiment_dir/efficiency_metrics.json."""
    path = experiment_dir / "efficiency_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Per-batch efficiency (no division by batch size)
# ---------------------------------------------------------------------------

def _encoder_time_per_batch(
    encoder: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_repeat: int = 100,
) -> float:
    """Encoder inference time for the batch (ms). No division by batch size."""
    encoder.eval()
    x = x.to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = encoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            _ = encoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / num_repeat) * 1000.0  # ms per batch


def _decoder_time_per_batch(
    decoder: nn.Module,
    decoder_inputs: Tuple[Any, ...],
    device: torch.device,
    num_warmup: int = 10,
    num_repeat: int = 100,
) -> float:
    """Decoder inference time for the batch (ms). No division by batch size."""
    decoder.eval()
    decoder_inputs = tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in decoder_inputs
    )
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = decoder(*decoder_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            _ = decoder(*decoder_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / num_repeat) * 1000.0  # ms per batch


def compute_efficiency_metrics_per_batch(
    model: nn.Module,
    model_type: str,
    sample_batch: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_repeat: int = 100,
) -> Dict[str, Any]:
    """
    Compute encoder/decoder parameter counts, inference time per batch (ms), and FLOPs per batch (total).
    No division by batch size; batch_size and (for beforeComp) num_windows are included for reference.

    Args:
        model: Full model (encoder + decoder).
        model_type: One of 'baseline', 'beforeComp', 'afterComp_v1', 'afterComp_v2'.
        sample_batch: One batch from the dataloader (e.g. shape (B, 2, 13, 32) or (B, T, 2, 13, 32)).
        device: Device to run on.
        num_warmup: Warmup iterations for timing.
        num_repeat: Repeat count for timing.

    Returns:
        Dict with encoder_params, decoder_params, encoder_time_ms_per_batch, decoder_time_ms_per_batch,
        encoder_flops_per_batch, decoder_flops_per_batch, batch_size, and (for beforeComp) num_windows.
    """
    actual = _get_actual_model(model)
    encoder = actual.encoder
    decoder = actual.decoder

    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)

    # Flatten time into batch if 5D (B, T, C, H, W) -> (B*T, C, H, W)
    if sample_batch.dim() == 5:
        sample_batch = sample_batch.reshape(
            -1, sample_batch.size(2), sample_batch.size(3), sample_batch.size(4)
        )

    # Ensure we have a batch on device for beforeComp (need enough frames for time_window)
    if model_type == "beforeComp" and sample_batch.dim() == 4:
        time_window = getattr(actual.encoder, "time_window", 10)
        if sample_batch.size(0) < time_window:
            n = (time_window + sample_batch.size(0) - 1) // sample_batch.size(0)
            sample_batch = sample_batch.repeat(n, 1, 1, 1)[:time_window]
    x = sample_batch.to(device)

    batch_size = x.size(0)

    # Encoder time (per batch)
    encoder_time_ms = _encoder_time_per_batch(encoder, x, device, num_warmup, num_repeat)

    try:
        decoder_inputs, effective_batch = _prepare_decoder_inputs(model_type, actual, x, device)
    except Exception as e:
        return {
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "encoder_time_ms_per_batch": round(encoder_time_ms, 4),
            "decoder_time_ms_per_batch": None,
            "encoder_flops_per_batch": None,
            "decoder_flops_per_batch": None,
            "batch_size": batch_size,
            "num_windows": None,
            "error": str(e),
        }

    # Decoder time (per batch)
    decoder_time_ms = _decoder_time_per_batch(decoder, decoder_inputs, device, num_warmup, num_repeat)

    # FLOPs (total for batch, no division)
    encoder_flops_total = _flops_encoder(encoder, x)
    decoder_flops_total = _flops_decoder(decoder, decoder_inputs, effective_batch)

    result = {
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "encoder_time_ms_per_batch": round(encoder_time_ms, 4),
        "decoder_time_ms_per_batch": round(decoder_time_ms, 4),
        "encoder_flops_per_batch": encoder_flops_total,
        "decoder_flops_per_batch": decoder_flops_total,
        "batch_size": batch_size,
    }
    if model_type == "beforeComp":
        result["num_windows"] = effective_batch
    else:
        result["num_windows"] = None
    return result
