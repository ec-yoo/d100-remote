"""The client package depends only on torch, torch-mlir, numpy and the Python standard library."""
import ast
import pathlib
import subprocess
import sys

PKG = pathlib.Path(__file__).resolve().parents[1] / "d100"
ALLOWED = {"torch", "torch_mlir", "numpy"} | set(sys.stdlib_module_names)


def test_source_imports_only_allowed_modules():
    bad = []
    for p in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            bad += [f"{p.name}: {n}" for n in names if n.split(".")[0] not in ALLOWED]
    assert not bad, bad


def test_import_loads_only_allowed_modules():
    # torch.compile pulls in torch's own dependencies (sympy, networkx, ...) lazily; load them first
    code = ("import sys, torch, numpy, torch._dynamo\n"
            "from torch._dynamo.backends.common import aot_autograd\n"
            "try:\n    import torch_mlir\nexcept ImportError:\n    pass\n"
            "before = {m.split('.')[0] for m in sys.modules}\n"
            "import d100.torch\n"
            "new = {m.split('.')[0] for m in sys.modules} - before - {'d100'} - set(sys.stdlib_module_names)\n"
            "assert not new, sorted(new)")
    r = subprocess.run([sys.executable, "-c", code], cwd=PKG.parent, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr



def test_a_new_input_size_compiles_with_fixed_shapes(monkeypatch):
    """Sizes stay plain integers when the same model is called with another input size."""
    import torch

    import d100.torch as backend

    seen = []

    def capture(gm, example_inputs):
        seen.append([tuple(a.shape) for a in example_inputs])
        return gm.forward

    monkeypatch.setattr(backend, "_compile_graph", capture)
    compiled = torch.compile(torch.nn.Conv2d(3, 4, 3), backend="d100")
    with torch.no_grad():
        compiled(torch.zeros(1, 3, 8, 8))
        compiled(torch.zeros(1, 3, 16, 16))
    assert len(seen) == 2 and all(type(d) is int for shapes in seen for shape in shapes for d in shape), seen
