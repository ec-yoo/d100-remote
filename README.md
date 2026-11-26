# d100-remote

PyTorch 모델을 HEAD D100 보드의 NPU 에서 돌리는 클라이언트 패키지 `d100` 이다. 자기 PC 에서 `torch.compile(model, backend="d100")` 한 줄을 쓰면, torch-mlir 가 모델을 StableHLO 로 낮추고(torch-neuronx 와 같은 앞단) 그 텍스트를 D100 API 서버로 보낸다. 서버가 컴파일하고 NPU 에서 실행해 결과를 돌려준다. 이 패키지는 torch, torch-mlir, numpy 만 쓴다. 컴파일러는 들어 있지 않다.

## 바로 확인

Python 3.10 이상이 필요하다. `<서버 주소>` 에 받은 D100 API 서버 주소를 넣고 아래를 붙여 넣는다. 마지막 줄에 `PASS` 가 나오면 설치, 서버 연결, NPU 실행이 모두 된 것이다.

```bash
git clone https://github.com/ec-yoo/d100-remote.git && cd d100-remote
python3 -m venv .venv && . .venv/bin/activate
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-mlir
pip install ./client
export D100_API_URL=http://<서버 주소>:8710
python check.py
```

`check.py` 는 ReLU 하나를 D100 NPU 에서 한 번 돌리고 PyTorch CPU 결과와 비교한다. `FAIL: 서버에 닿지 않는다` 가 나오면 `curl -s $D100_API_URL/v1/health` 로 서버 응답부터 확인한다.

## 설치

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-mlir
pip install ./client
```

서버 주소는 `D100_API_URL` 로 준다. 기본값은 없다.

## 사용

```python
import torch, torch.nn as nn
import d100.torch                      # "d100" 백엔드 등록

model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
with torch.no_grad():
    y = torch.compile(model, backend="d100")(torch.randn(2, 16))   # D100 NPU 에서 계산
```

- 추론만 된다. `torch.no_grad()` 안에서 부른다.
- 어떤 PyTorch 연산이든 torch-mlir 가 StableHLO 로 낮출 수 있으면 보낼 수 있다. 서버가 아직 모르는 StableHLO 연산(예: `sort`, `while`)이나 torch-mlir 가 못 낮추는 연산은 첫 호출 때 오류가 연산 이름과 함께 난다.
- `D100_API_URL` 로 서버 주소(필수)를, `D100_API_KEY` 로 API 키를 준다.
- 첫 호출은 StableHLO 변환, 가중치 업로드(해시로 한 번만), 서버 컴파일로 시간이 걸린다. 다음 호출부터는 입력과 결과만 오간다.
- 모델 안에 torch.compile 이 그래프를 끊는 지점(데이터에 따라 달라지는 연산 등)이 있으면 조각마다 따로 보낸다. 결과는 같고 왕복이 늘어난다.

## 파일

| 경로 | 하는 일 |
|---|---|
| `client/d100/torch.py` | `torch.compile` 백엔드 `"d100"` 등록 |
| `client/d100/_lower.py` | torch.compile 이 잡은 그래프를 torch-mlir 로 StableHLO 텍스트로 낮춘다. 매개변수는 입력으로 남겨 텍스트에 넣지 않는다. torch-mlir 가 못 낮추거나 PyTorch 와 뜻이 다르게 낮추는 연산(nearest upsample, remainder)은 여기서 기본 연산으로 먼저 쪼갠다 |
| `client/d100/remote.py` | HTTP 클라이언트. 모듈은 텍스트의 SHA-256, 가중치는 `.npy` 바이트의 SHA-256 으로 한 번씩만 올린다 |
| `client/tests/test_ops.py` | 레이어·함수 30종이 PyTorch CPU 와 같은지, 서버가 모르는 연산은 첫 호출에서 오류가 나는지(서버가 안 닿으면 건너뜀) |
| `client/tests/test_package.py` | 패키지가 torch, torch-mlir, numpy, 표준 라이브러리 밖의 모듈을 import 하지 않는지 |

## 확인

```bash
cd client && python -m pytest tests -q
curl -s $D100_API_URL/v1/health     # {"ok": true, ...}
```

서버 코드는 이 저장소에 없다.
