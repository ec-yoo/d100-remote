"""PyTorch entry: `torch.compile(model, backend="d100")` runs the model on a D100 board through the API server.

    import d100.torch   # registers the "d100" backend
    compiled = torch.compile(model, backend="d100")
    with torch.no_grad():
        y = compiled(x)

On the first call the graph that torch.compile captures is lowered to StableHLO with torch-mlir (the
front end torch-neuronx also uses) and sent to the server, which compiles it for the NPU. Later calls
send only the inputs. Parameters are uploaded once per process, identified by content hash.
Inference only: calling backward() raises D100UnsupportedOp.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd

from ._lower import D100UnsupportedOp, to_stablehlo
from .remote import Session, encode_input

__all__ = ["D100UnsupportedOp"]

# The device runs fixed shapes. By default torch.compile turns a size into a symbolic integer once it sees that
# size change, and a graph with symbolic sizes cannot be lowered: a new size gets its own compiled graph instead.
if hasattr(torch._dynamo.config, "automatic_dynamic_shapes"):
    torch._dynamo.config.automatic_dynamic_shapes = False


def _compile_graph(gm: torch.fx.GraphModule, example_inputs) -> Callable:
    if any(isinstance(a, torch.SymInt) for a in example_inputs):
        raise D100UnsupportedOp('dynamic shapes are not supported: use torch.compile(model, backend="d100", dynamic=False)')
    mlir = to_stablehlo(gm, example_inputs)  # raises D100UnsupportedOp now, not at the first call
    session = Session()
    module = session.compile(mlir)

    def run(*args):
        inputs = [session.weight_ref(a) if isinstance(a, torch.nn.Parameter) else encode_input(a) for a in args]
        return session.run(module, inputs)

    run._boxed_call = False
    return run


def _backend(gm, example_inputs, options: dict | None = None):
    opts = dict(options or {})
    if opts.pop("target", "remote") != "remote":
        raise ValueError("this d100 package runs models on the D100 API server only (target='remote')")
    if opts:
        raise ValueError(f"unknown d100 options: {sorted(opts)}")

    def fw(fx_gm, inputs):
        return _compile_graph(fx_gm, inputs)

    def bw(_fx_gm, _inputs):
        raise D100UnsupportedOp("D100 backend runs inference only; call the model under torch.no_grad()")

    return aot_autograd(fw_compiler=fw, bw_compiler=bw)(gm, example_inputs)


register_backend(_backend, name="d100")
