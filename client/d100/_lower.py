"""Lower an aten FX graph to StableHLO text with torch-mlir (the same front end torch-neuronx uses).

`torch.compile` hands the backend an FX graph whose inputs are the model's parameters and the call's
tensors, in order. That graph is exported, decomposed to the core ATen operator set, and lowered by
torch-mlir to StableHLO. Parameters stay inputs (they are not baked into the text), so the weights travel
separately and only once.
"""
from __future__ import annotations

import torch

aten = torch.ops.aten


class D100UnsupportedOp(NotImplementedError):
    """The model uses an operator that cannot be lowered for the D100 backend."""


def _upsample_nearest(x, output_size=None, *_, **__):
    # torch-mlir has no StableHLO lowering for nearest upsampling; an integer scale is a broadcast + reshape.
    n, c, h, w = x.shape
    oh, ow = (int(v) for v in output_size)
    if oh % h or ow % w:
        raise D100UnsupportedOp(f"upsample_nearest2d with non-integer scale ({h}x{w} -> {oh}x{ow})")
    sh, sw = oh // h, ow // w
    return x[:, :, :, None, :, None].expand(n, c, h, sh, w, sw).reshape(n, c, oh, ow)


def _upsample_nearest_vec(x, output_size, scale_factors):
    if output_size is None:
        h, w = x.shape[-2:]
        output_size = [int(h * scale_factors[0]), int(w * scale_factors[1])]
    return _upsample_nearest(x, output_size)


def _remainder(a, b):
    # torch.remainder follows Python (sign of the divisor); torch-mlir lowers it to stablehlo.remainder,
    # which is C fmod (sign of the dividend) and differs for negative inputs.
    return a - torch.floor(a / b) * b


def decompositions() -> dict:
    from torch._decomp import core_aten_decompositions

    table = core_aten_decompositions()
    table[aten.upsample_nearest2d.default] = _upsample_nearest
    table[aten.upsample_nearest2d.vec] = _upsample_nearest_vec
    table[aten.remainder.Tensor] = _remainder
    table[aten.remainder.Scalar] = _remainder
    return table


def to_stablehlo(gm: torch.nn.Module, example_inputs) -> str:
    """StableHLO module text for `gm` called with `example_inputs` (all inputs become `@main` arguments)."""
    try:
        from torch_mlir import fx
    except ImportError as e:
        raise ImportError("the d100 backend needs torch-mlir: pip install torch-mlir") from e

    try:
        ep = torch.export.export(gm, tuple(example_inputs), strict=False)
        ep = ep.run_decompositions(decompositions())
        return str(fx.export_and_import(ep, output_type="stablehlo"))
    except D100UnsupportedOp:
        raise
    except Exception as e:  # torch-mlir reports an operator it cannot legalize
        msg = str(e).strip().splitlines()
        raise D100UnsupportedOp("could not lower the model to StableHLO: " + " | ".join(msg[:3])) from e
