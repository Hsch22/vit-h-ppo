"""Persistent cross-environment pool of SONIC/MuJoCo tracker workers.

Allows the PPO trainer running inside `vith-base` to evaluate batches of G1
trajectories in parallel inside the isolated `vith-tracker` Python environment
with zero per-step process spawn and zero disk I/O.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pickle
import struct
import subprocess
import threading
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = Path(__file__).resolve().with_name("tracker_worker.py")


def _read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)


class _WorkerHandle:
    def __init__(self, cmd: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.lock = threading.Lock()
        ready = self._recv()
        if not isinstance(ready, dict) or not ready.get("ready"):
            err = self.proc.stderr.read().decode("utf-8", errors="ignore") if self.proc.stderr else ""
            self.close()
            raise RuntimeError(f"Tracker worker failed to initialize: {ready!r}\n{err}")
        self.protocol_version = int(ready["tracker_protocol_version"])

    def _send(self, obj: object) -> None:
        assert self.proc.stdin is not None
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.proc.stdin.write(struct.pack("!I", len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def _recv(self) -> Any:
        assert self.proc.stdout is not None
        header = _read_exact(self.proc.stdout, 4)
        if len(header) != 4:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore") if self.proc.stderr else ""
            raise RuntimeError(f"Tracker worker closed unexpectedly: {err[:400]}")
        length = struct.unpack("!I", header)[0]
        raw = _read_exact(self.proc.stdout, length)
        if len(raw) != length:
            raise RuntimeError("Truncated response from tracker worker")
        return pickle.loads(raw)

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._send(payload)
            return self._recv()

    def close(self) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.write(struct.pack("!I", 0))
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class PersistentTrackerPool:
    """Thread-safe pool of long-lived `vith-tracker` worker subprocesses."""

    def __init__(
        self,
        tracker_python: Path | str,
        policy_onnx: Path | str,
        xml_path: Path | str,
        grounding_xml: Path | str,
        num_workers: int = 8,
        control_hz: float = 50.0,
        ort_threads: int = 1,
    ) -> None:
        if int(num_workers) < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self.tracker_python = Path(tracker_python).expanduser().resolve()
        self.policy_onnx = Path(policy_onnx).expanduser().resolve()
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.grounding_xml = Path(grounding_xml).expanduser().resolve()
        for p in (self.tracker_python, self.policy_onnx, self.xml_path, self.grounding_xml):
            if not p.is_file():
                raise FileNotFoundError(f"Missing tracker pool asset: {p}")

        cmd = [
            str(self.tracker_python),
            "-u",
            str(WORKER_SCRIPT),
            "--policy-onnx",
            str(self.policy_onnx),
            "--xml-path",
            str(self.xml_path),
            "--grounding-xml",
            str(self.grounding_xml),
            "--control-hz",
            str(control_hz),
            "--ort-threads",
            str(ort_threads),
        ]
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"

        self.num_workers = int(num_workers)
        with ThreadPoolExecutor(max_workers=self.num_workers) as init_pool:
            self._workers = list(init_pool.map(lambda _: _WorkerHandle(cmd, env), range(self.num_workers)))
        self.protocol_version = self._workers[0].protocol_version
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def evaluate_batch(
        self,
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        body_pos_w: np.ndarray,
        lengths: np.ndarray | list[int],
        fps: float | np.ndarray | list[float] = 30.0,
    ) -> list[dict[str, Any]]:
        """Evaluate S trajectories in parallel; preserves exact sample order 0..S-1."""
        samples = int(joint_pos.shape[0])
        if np.ndim(fps) == 0:
            fps_arr = [float(fps)] * samples
        else:
            fps_arr = [float(x) for x in fps]
            if len(fps_arr) != samples:
                raise ValueError(f"Expected {samples} fps values, got {len(fps_arr)}")
        jobs = []
        for idx in range(samples):
            valid_len = int(lengths[idx])
            payload = {
                "joint_pos": np.ascontiguousarray(joint_pos[idx, :valid_len], dtype=np.float32),
                "root_pos_w": np.ascontiguousarray(root_pos_w[idx, :valid_len], dtype=np.float32),
                "root_quat_w": np.ascontiguousarray(root_quat_w[idx, :valid_len], dtype=np.float32),
                "body_pos_w": np.ascontiguousarray(body_pos_w[idx, :valid_len], dtype=np.float32),
                "fps": fps_arr[idx],
            }
            worker = self._workers[idx % self.num_workers]
            jobs.append(self._executor.submit(worker.evaluate, payload))
        return [job.result() for job in jobs]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for worker in self._workers:
            worker.close()
        self._workers.clear()

    def __enter__(self) -> "PersistentTrackerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
