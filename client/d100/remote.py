"""HTTP client for the D100 API server.

A compiled graph is a StableHLO module, identified by the SHA-256 of its text: POST /v1/compile once, then
POST /v1/run with the module hash and the inputs. Weights travel once: each one is identified by the SHA-256 of its `.npy` bytes, uploaded with PUT when the
server does not have it yet, and referenced by hash afterwards.

    D100_API_URL   server base URL, for example http://<server>:8710 (required)
    D100_API_KEY   sent as "Authorization: Bearer <key>" when set
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def array_to_b64(a: np.ndarray) -> str:
    return base64.b64encode(npy_bytes(a)).decode("ascii")


def b64_to_array(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


class D100RemoteError(RuntimeError):
    """The D100 API server rejected the request or failed while running it."""


def api_url() -> str:
    url = os.environ.get("D100_API_URL")
    if not url:
        raise D100RemoteError("D100_API_URL is not set: export D100_API_URL=http://<server>:8710")
    return url.rstrip("/")


def request(method: str, path: str, body: bytes | None = None, content_type: str = "application/json",
            timeout: float = 3600.0) -> tuple[int, bytes]:
    req = urllib.request.Request(api_url() + path, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", content_type)
    if key := os.environ.get("D100_API_KEY"):
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def health(timeout: float = 5.0) -> dict | None:
    """Server status (`{"ok": true, "busy": ..., "board_runs": ...}`), or None when unreachable."""
    try:
        status, body = request("GET", "/v1/health", timeout=timeout)
    except (OSError, D100RemoteError):
        return None
    return json.loads(body) if status == 200 else None


class Session:
    """Remembers which weights the server already holds, so each is hashed and uploaded once."""

    def __init__(self) -> None:
        self._sha: dict[int, tuple[Any, int, str]] = {}  # id(tensor) -> (tensor, version, sha)
        self._uploaded: set[str] = set()

    def weight_ref(self, t) -> dict:
        """`{"$w": sha}` for a tensor that stays the same across calls (a parameter or constant)."""
        version = getattr(t, "_version", 0)
        hit = self._sha.get(id(t))
        if hit is None or hit[0] is not t or hit[1] != version:
            data = npy_bytes(t.detach().cpu().numpy())
            sha = hashlib.sha256(data).hexdigest()
            if sha not in self._uploaded:
                status, _ = request("HEAD", f"/v1/tensor/{sha}")
                if status != 200:
                    status, out = request("PUT", f"/v1/tensor/{sha}", data, "application/octet-stream")
                    if status not in (200, 201):
                        raise D100RemoteError(f"weight upload failed ({status}): {out[:500]!r}")
                self._uploaded.add(sha)
            hit = (t, version, sha)
            self._sha[id(t)] = hit
        return {"$w": hit[2]}

    def compile(self, mlir: str) -> str:
        """Hand a StableHLO module to the server (once per content) and return its id."""
        sha = hashlib.sha256(mlir.encode()).hexdigest()
        status, _ = request("HEAD", f"/v1/module/{sha}")
        if status != 200:
            status, out = request("POST", "/v1/compile", gzip.compress(mlir.encode()), "application/gzip")
            msg = _json(status, out)
            if status != 200:
                raise D100RemoteError(f"compile failed ({status}): {msg.get('error', msg)}")
            if msg["module"] != sha:
                raise D100RemoteError("server stored the module under a different hash")
        return sha

    def run(self, module: str, inputs: list) -> Any:
        body = json.dumps({"module": module, "inputs": inputs}).encode()
        status, out = request("POST", "/v1/run", body)
        msg = _json(status, out)
        if status != 200:
            raise D100RemoteError(f"{status}: {msg.get('error', msg)}")
        return decode_outputs(msg["outputs"])


def _json(status: int, out: bytes) -> dict:
    try:
        return json.loads(out)
    except ValueError:
        raise D100RemoteError(f"server answered {status} with non-JSON: {out[:300]!r}") from None


def encode_input(a) -> Any:
    import torch

    if isinstance(a, torch.Tensor):
        return {"$a": array_to_b64(a.detach().cpu().numpy())}
    return a


def decode_outputs(v):
    if isinstance(v, list):
        return [decode_outputs(x) for x in v]
    if isinstance(v, dict) and "$t" in v:
        return tuple(decode_outputs(x) for x in v["$t"])
    if isinstance(v, dict) and "$a" in v:
        import torch

        return torch.from_numpy(b64_to_array(v["$a"]))
    return v
