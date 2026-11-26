"""D100 원격 실행이 되는지 한 번에 확인한다:  python check.py

ReLU 하나를 D100 보드의 NPU 에서 한 번 돌리고 PyTorch CPU 결과와 비교한다.
마지막 줄이 PASS 면 설치, 서버 연결, NPU 실행이 모두 된 것이다.
"""
import os
import sys
import time

import torch
import torch.nn as nn

import d100.torch  # noqa: F401 - "d100" 백엔드 등록
from d100.remote import api_url, health

if not os.environ.get("D100_API_URL"):
    sys.exit("FAIL: D100_API_URL 이 비어 있다. export D100_API_URL=http://<서버 주소>:8710 으로 지정한다.")
status = health()
print(f"서버 {api_url()}: {status}")
if status is None:
    sys.exit("FAIL: 서버에 닿지 않는다. D100_API_URL 주소와 네트워크 연결을 확인한다.")

torch.manual_seed(0)
model = nn.ReLU()
x = torch.randn(4, 8)

start = time.time()
with torch.no_grad():
    got = torch.compile(model, backend="d100")(x)
error = float((got - model(x)).abs().max())
print(f"ReLU 를 보드에서 실행: {time.time() - start:.2f} s, PyTorch CPU 와의 최대 오차 {error:.1e}")
print("PASS" if error < 1e-5 else "FAIL: 결과가 PyTorch 와 다르다")
sys.exit(0 if error < 1e-5 else 1)
