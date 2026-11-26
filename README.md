# d100-remote

PC의 PyTorch 모델을 `torch.compile` 한 줄로 HEAD D100 NPU에서 실행합니다.

```python
import d100.torch
compiled = torch.compile(model, backend="d100")
with torch.no_grad():
    y = compiled(x)          # D100 NPU에서 실행
```

`d100` 패키지는 `torch.compile`이 잡은 그래프를 torch-mlir로 StableHLO로 변환해 D100 API 서버로 보냅니다. 서버가 NPU용으로 컴파일하고 실행한 뒤 결과를 PyTorch 텐서로 돌려줍니다. 보드에는 아무것도 설치하지 않으며, 패키지에는 컴파일러가 들어 있지 않습니다. 의존성은 torch, torch-mlir, numpy뿐입니다.

## 요구 사항

- Linux (x86_64), Python 3.10 이상
- D100 API 서버 주소. 서버가 요구하면 API 키

Python 3.11.15, torch 2.15.0+cpu, torch-mlir 20261101에서 확인했습니다.

## 빠른 시작

```bash
git clone https://github.com/ec-yoo/d100-remote.git
cd d100-remote
python3 -m venv .venv && . .venv/bin/activate
pip install numpy torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-mlir==20261101
pip install ./client

export D100_API_URL=http://<서버 주소>:8710
export D100_API_KEY=<API 키>      # 키를 받은 경우만
python check.py
```

`check.py`는 ReLU 하나를 NPU에서 실행하고 PyTorch CPU 결과와 비교합니다. 마지막 줄이 `PASS`이면 설치, 서버 연결, NPU 실행이 모두 정상입니다.

```text
서버 http://<서버 주소>:8710: {'ok': True, 'busy': False}
ReLU 를 보드에서 실행: 8.65 s, PyTorch CPU 와의 최대 오차 0.0e+00
PASS
```

| 출력 | 원인 |
|---|---|
| `FAIL: D100_API_URL 이 비어 있다` | 이 터미널에서 `D100_API_URL`을 지정하지 않았습니다 |
| `FAIL: 서버에 닿지 않는다` | 서버에 연결되지 않았거나 서버가 요청을 거부했습니다. `curl -i $D100_API_URL/v1/health`로 확인하고, `401`이면 API 키를 확인합니다 |

## 사용 시 참고

- 추론만 실행합니다. 모델은 `torch.no_grad()` 안에서 호출하며, `backward()`를 부르면 `D100UnsupportedOp`가 납니다.
- 입력 모양마다 한 번씩 컴파일합니다. 새 모양의 첫 호출은 오래 걸리고, 이후 호출은 입력만 보냅니다. 가중치는 내용 해시로 구분해 한 번만 올립니다.
- torch-mlir가 변환하지 못하는 연산은 첫 호출에서 `D100UnsupportedOp`를 냅니다. 서버가 아직 지원하지 않는 연산(예: `sort`, `while`)은 `D100RemoteError: compile failed`를 냅니다. 두 메시지 모두 연산 이름을 포함합니다.
- `torch.compile`이 모델을 여러 그래프로 나누면 조각마다 따로 NPU에서 실행합니다. 나뉘는 지점은 `fullgraph=True`로 찾습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `client/d100/torch.py` | `torch.compile` 백엔드 `"d100"` 등록 |
| `client/d100/_lower.py` | 잡은 그래프를 torch-mlir로 StableHLO로 변환. 매개변수는 그래프 입력으로 남겨 가중치를 따로 보냄 |
| `client/d100/remote.py` | D100 API 서버 HTTP 클라이언트 |
| `client/tests/` | 서버 대상 연산 시험(`D100_API_URL`이 없으면 건너뜀)과 패키지 의존성 시험 |
| `check.py` | 설치 확인 스크립트 |

## 시험

```bash
cd client && python -m pytest tests -q
```

## 라이선스

Apache License 2.0. `LICENSE` 파일을 참고합니다.
