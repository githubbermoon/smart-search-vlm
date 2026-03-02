from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import StackConfig

_LAST_SPAWN_TS: float = 0.0
_SPAWN_COOLDOWN_SEC = 1.5


def _pid_file_path(cfg: StackConfig) -> Path:
    return Path(f"{cfg.text_embed_socket_path}.pid")


def _is_pid_alive(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
        import os

        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _daemon_running(cfg: StackConfig) -> bool:
    pid_file = _pid_file_path(cfg)
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    return _is_pid_alive(pid)


def _request_vector(
    *,
    socket_path: str,
    model_name: str,
    text: str,
    is_query: bool,
    timeout_sec: float,
) -> list[float] | None:
    if not Path(socket_path).exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_sec)
            sock.connect(socket_path)
            payload = {
                "text": text,
                "is_query": is_query,
                "model_name": model_name,
            }
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            max_bytes = 8 * 1024 * 1024
            total = 0
            while total < max_bytes:
                chunk = sock.recv(64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if b"\n" in chunk:
                    break
            if not chunks:
                return None
            raw = b"".join(chunks)
            line = raw.decode("utf-8", errors="ignore").splitlines()[0]
            data = json.loads(line)
            if not bool(data.get("ok")):
                return None
            vec = data.get("vector")
            if not isinstance(vec, list):
                return None
            return [float(x) for x in vec]
    except Exception:
        return None


def _spawn_daemon(cfg: StackConfig) -> None:
    global _LAST_SPAWN_TS
    if _daemon_running(cfg):
        return
    now = time.time()
    if (now - _LAST_SPAWN_TS) < _SPAWN_COOLDOWN_SEC:
        return
    _LAST_SPAWN_TS = now

    cmd = [
        sys.executable,
        "-m",
        "mm_stack.text_embed_daemon",
        "--socket-path",
        cfg.text_embed_socket_path,
        "--model-name",
        cfg.text_model_name,
        "--pid-file",
        str(_pid_file_path(cfg)),
    ]
    subprocess.Popen(
        cmd,
        cwd=str(cfg.stack_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def encode_with_daemon(
    cfg: StackConfig,
    *,
    text: str,
    is_query: bool = True,
) -> list[float] | None:
    vector = _request_vector(
        socket_path=cfg.text_embed_socket_path,
        model_name=cfg.text_model_name,
        text=text,
        is_query=is_query,
        timeout_sec=0.5,
    )
    if vector is not None:
        return vector

    if not cfg.text_embed_daemon_autostart:
        return None

    _spawn_daemon(cfg)
    deadline = time.time() + max(0.5, cfg.text_embed_daemon_start_timeout_ms / 1000.0)
    while time.time() < deadline:
        vector = _request_vector(
            socket_path=cfg.text_embed_socket_path,
            model_name=cfg.text_model_name,
            text=text,
            is_query=is_query,
            timeout_sec=0.5,
        )
        if vector is not None:
            return vector
        time.sleep(0.05)
    return None
