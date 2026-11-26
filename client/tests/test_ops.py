"""Each PyTorch layer or function, run on the D100 board through the API server, must match PyTorch CPU.

Skipped when the server (D100_API_URL) does not answer.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import d100.torch  # noqa: F401
from d100.remote import health


class Fn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *xs):
        return self.fn(*xs)


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, q):
        return self.attn(q, q, q)[0]


torch.manual_seed(0)
CASES = {
    "Linear": (nn.Linear(16, 8), [torch.randn(3, 16)]),
    "Conv2d": (nn.Conv2d(3, 4, 3, stride=1, padding=1), [torch.randn(1, 3, 8, 8)]),
    "Conv2d stride 2": (nn.Conv2d(3, 4, 3, stride=2, padding=1), [torch.randn(1, 3, 8, 8)]),
    "Conv2d groups": (nn.Conv2d(4, 4, 3, padding=1, groups=4), [torch.randn(1, 4, 8, 8)]),
    "ReLU": (nn.ReLU(), [torch.randn(4, 8)]),
    "Sigmoid": (nn.Sigmoid(), [torch.randn(4, 8)]),
    "Tanh": (nn.Tanh(), [torch.randn(4, 8)]),
    "SiLU": (nn.SiLU(), [torch.randn(4, 8)]),
    "GELU": (nn.GELU(), [torch.randn(4, 8)]),
    "Softmax": (nn.Softmax(dim=-1), [torch.randn(4, 8)]),
    "LogSoftmax": (nn.LogSoftmax(dim=-1), [torch.randn(4, 8)]),
    "MaxPool2d": (nn.MaxPool2d(2), [torch.randn(1, 3, 8, 8)]),
    "AvgPool2d": (nn.AvgPool2d(2), [torch.randn(1, 3, 8, 8)]),
    "Flatten": (nn.Flatten(), [torch.randn(2, 3, 4, 4)]),
    "add sub mul div": (Fn(lambda a, b: (a + b) * a - b / (b.abs() + 1)), [torch.randn(4, 8), torch.randn(4, 8)]),
    "abs neg exp sqrt": (Fn(lambda x: (-x).exp() + x.abs().sqrt()), [torch.randn(4, 8)]),
    "matmul": (Fn(torch.matmul), [torch.randn(4, 8), torch.randn(8, 5)]),
    "bmm": (Fn(torch.bmm), [torch.randn(2, 4, 8), torch.randn(2, 8, 5)]),
    "permute transpose": (Fn(lambda x: x.permute(0, 2, 1).transpose(1, 2)), [torch.randn(2, 3, 4)]),
    "reshape unsqueeze squeeze": (Fn(lambda x: x.reshape(2, 12).unsqueeze(1).squeeze(1)), [torch.randn(2, 3, 4)]),
    "cat": (Fn(lambda a, b: torch.cat([a, b], dim=1)), [torch.randn(2, 3), torch.randn(2, 4)]),
    "mean sum": (Fn(lambda x: x.mean(dim=1) + x.sum(dim=1)), [torch.randn(4, 8)]),
    "F.relu F.softmax": (Fn(lambda x: F.softmax(F.relu(x), dim=-1)), [torch.randn(4, 8)]),
    "Upsample": (nn.Upsample(scale_factor=2), [torch.randn(1, 3, 4, 4)]),
    "LayerNorm": (nn.LayerNorm(8), [torch.randn(2, 3, 8)]),
    "Embedding": (nn.Embedding(10, 6), [torch.tensor([[1, 3, 9], [0, 2, 2]])]),
    "ConvTranspose2d": (nn.ConvTranspose2d(3, 2, 2, stride=2), [torch.randn(1, 3, 4, 4)]),
    "MultiheadAttention": (SelfAttention(), [torch.randn(1, 5, 8)]),
    "pad flip cumsum": (Fn(lambda x: F.pad(x, (1, 2), value=0.5).flip(-1).cumsum(-1)), [torch.randn(3, 5)]),
    "sort topk": (Fn(lambda x: torch.sort(x, -1).values + torch.topk(x, 5, -1).values.flip(-1)), [torch.randn(4, 5)]),
    "scatter_add index_put": (Fn(lambda x, i, s: x.scatter_add(1, i, s).index_put((torch.tensor([2]),), torch.ones(5))),
                              [torch.zeros(3, 5), torch.tensor([[0, 1, 4], [2, 2, 0], [3, 1, 1]]), torch.randn(3, 3)]),
    "argmax remainder round": (Fn(lambda x: x.argmax(-1).float() + torch.remainder(x * 3, 2.0).sum(-1) + (x * 4).round().sum(-1)), [torch.randn(4, 7)]),
}

# Cases whose result goes through a matmul or convolution, which the NPU computes at reduced precision.
QUANTIZED = {"Linear", "Conv2d", "Conv2d stride 2", "Conv2d groups", "matmul", "bmm", "ConvTranspose2d",
             "MultiheadAttention", "Flatten"}

pytestmark = pytest.mark.skipif(health() is None, reason="D100 API server not reachable (D100_API_URL)")


@pytest.mark.parametrize("name", list(CASES))
def test_op(name):
    module, inputs = CASES[name]
    module = module.eval()
    torch._dynamo.reset()
    with torch.no_grad():
        ref = module(*inputs)
        got = torch.compile(module, backend="d100")(*inputs)
    assert got.shape == ref.shape
    err = float((got - ref).abs().max())
    if name in QUANTIZED:
        assert err <= 0.05 * float(ref.abs().max()), (err, float(ref.abs().max()))
    else:
        assert err <= 1e-4 + 1e-4 * float(ref.abs().max()), err


def test_unlowerable_op_fails_at_compile_time():
    # torch.fft has complex numbers, which torch-mlir does not lower to StableHLO: the error comes on the
    # first call, before anything is sent to the board.
    torch._dynamo.reset()
    with torch.no_grad(), pytest.raises(Exception, match="could not lower"):
        torch.compile(Fn(lambda x: torch.fft.rfft(x).abs()), backend="d100")(torch.randn(8))
