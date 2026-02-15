#!/usr/bin/env python3
"""Run a single-image Smart Stack feasibility test with RAM/swap monitoring."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_IMAGE = "/Users/pranjal/garage/smart_stack/processed/spiritual_art.jpg"
INGEST_PATH = Path("/Users/pranjal/garage/smart_stack/ingest.py")
DB_PATH = Path("/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db")


@dataclass
class Sample:
    ts_utc: str
    elapsed_s: float
    proc_rss_mb: float
    proc_cpu_pct: float
    sys_used_mb: float
    sys_free_mb: float
    swap_used_mb: float


def run_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def parse_size_to_mb(text: str) -> float:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", text)
    if not m:
        return 0.0
    value = float(m.group(1))
    unit = m.group(2)
    factors = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0, "P": 1024.0 * 1024.0 * 1024.0}
    return value * factors.get(unit, 1.0)


def get_process_metrics(pid: int) -> tuple[float, float]:
    rss_kb = run_output(["ps", "-o", "rss=", "-p", str(pid)])
    cpu_pct = run_output(["ps", "-o", "%cpu=", "-p", str(pid)])
    rss_mb = float(rss_kb) / 1024.0 if rss_kb else 0.0
    return rss_mb, float(cpu_pct or 0.0)


def get_system_memory_mb() -> tuple[float, float]:
    vm = run_output(["vm_stat"])
    page_size_match = re.search(r"page size of\s+(\d+)\s+bytes", vm)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096

    pages: dict[str, int] = {}
    for line in vm.splitlines():
        m = re.match(r"Pages\s+([^:]+):\s+([0-9]+)\.", line.strip())
        if m:
            pages[m.group(1).strip().lower()] = int(m.group(2))

    active = pages.get("active", 0)
    inactive = pages.get("inactive", 0)
    speculative = pages.get("speculative", 0)
    wired = pages.get("wired down", 0)
    compressed = pages.get("occupied by compressor", 0)
    free = pages.get("free", 0)

    used_pages = active + inactive + speculative + wired + compressed
    used_mb = (used_pages * page_size) / (1024.0 * 1024.0)
    free_mb = (free * page_size) / (1024.0 * 1024.0)
    return used_mb, free_mb


def get_swap_used_mb() -> float:
    swap = run_output(["sysctl", "vm.swapusage"])
    m = re.search(r"used\s*=\s*([0-9.]+[KMGTP])", swap)
    if not m:
        return 0.0
    return parse_size_to_mb(m.group(1))


def load_ingest_module() -> Any:
    spec = importlib.util.spec_from_file_location("ingest", INGEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {INGEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def monitor(pid: int, start: float, interval_s: float, samples: list[Sample], stop: threading.Event) -> None:
    while not stop.is_set():
        now = time.time()
        rss_mb, cpu_pct = get_process_metrics(pid)
        used_mb, free_mb = get_system_memory_mb()
        swap_used_mb = get_swap_used_mb()
        samples.append(
            Sample(
                ts_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_s=now - start,
                proc_rss_mb=rss_mb,
                proc_cpu_pct=cpu_pct,
                sys_used_mb=used_mb,
                sys_free_mb=free_mb,
                swap_used_mb=swap_used_mb,
            )
        )
        time.sleep(interval_s)


def fetch_latest_row(filename: str) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "select filename, caption, tags, ocr_text, processed_at from processed_images where filename = ? order by processed_at desc limit 1",
            (filename,),
        ).fetchone()
        return dict(row) if row else None


def write_csv(path: Path, samples: list[Sample]) -> None:
    if not samples:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(samples[0]).keys()))
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feasibility test runner for Qwen3-VL-8B in Smart Stack")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model id (default: {DEFAULT_EMBED_MODEL})")
    p.add_argument("--image", default=DEFAULT_IMAGE, help=f"Image path for single-file run (default: {DEFAULT_IMAGE})")
    p.add_argument("--sample-interval", type=float, default=1.0, help="Sampling interval in seconds")
    p.add_argument("--skip-download", action="store_true", help="Skip model snapshot download check")
    p.add_argument("--output-dir", default="/Users/pranjal/garage/smart_stack/feasibility_logs", help="Directory for json/csv logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = Path(args.image)
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    download_start = time.time()
    snapshot_path = ""
    if not args.skip_download:
        snapshot_path = snapshot_download(repo_id=args.model, repo_type="model")
    download_elapsed = time.time() - download_start

    ingest = load_ingest_module()

    samples: list[Sample] = []
    stop = threading.Event()
    mon = threading.Thread(target=monitor, args=(os.getpid(), time.time(), args.sample_interval, samples, stop), daemon=True)
    mon.start()

    success = False
    err = ""
    try:
        ingest.ensure_dirs()
        db = ingest.build_sqlite()
        from mlx_vlm import load
        from sentence_transformers import SentenceTransformer

        print(f"[RUN] model={args.model}")
        print(f"[RUN] embed_model={args.embed_model}")
        print(f"[RUN] image={image}")
        model, processor = load(args.model)
        embedder = SentenceTransformer(
            args.embed_model,
            trust_remote_code=ingest.requires_trust_remote_code(args.embed_model),
        )
        vector_table_name = ingest.table_for_embed_model(args.embed_model)
        try:
            ingest.process_one_image(
                db=db,
                file_path=image,
                vlm_model=model,
                vlm_processor=processor,
                embedder=embedder,
                embed_model_id=args.embed_model,
                vector_table_name=vector_table_name,
                safe_reprocess=True,
                print_fields=True,
            )
            success = True
        finally:
            del model
            del processor
            del embedder
    except Exception as e:
        err = str(e)
    finally:
        stop.set()
        mon.join(timeout=5)

    total_elapsed = time.time() - start
    filename = image.name
    latest = fetch_latest_row(filename)

    peak_proc_rss = max((s.proc_rss_mb for s in samples), default=0.0)
    peak_sys_used = max((s.sys_used_mb for s in samples), default=0.0)
    peak_swap = max((s.swap_used_mb for s in samples), default=0.0)

    report = {
        "run_ts_utc": run_ts,
        "model": args.model,
        "embed_model": args.embed_model,
        "vector_table": ingest.table_for_embed_model(args.embed_model),
        "image": str(image),
        "snapshot_path": snapshot_path,
        "download_elapsed_s": round(download_elapsed, 3),
        "total_elapsed_s": round(total_elapsed, 3),
        "success": success,
        "error": err,
        "sample_count": len(samples),
        "peaks": {
            "proc_rss_mb": round(peak_proc_rss, 2),
            "sys_used_mb": round(peak_sys_used, 2),
            "swap_used_mb": round(peak_swap, 2),
        },
        "db_latest_row": latest,
    }

    json_path = output_dir / f"qwen3_vl8b_feasibility_{run_ts}.json"
    csv_path = output_dir / f"qwen3_vl8b_feasibility_{run_ts}.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, samples)

    print(f"[REPORT] {json_path}")
    print(f"[SAMPLES] {csv_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
