#!/usr/bin/env python3
"""Extract immutable performance traces from browser benchmark JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_trace(log_path: Path) -> tuple[str, dict]:
    completed = None
    with log_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{log_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if record.get("event") == "done":
                completed = record
    if completed is None:
        raise ValueError(f"{log_path}: no completed benchmark record")
    sid = str(completed.get("sid") or "").strip()
    trace = completed.get("trace")
    if not sid or not isinstance(trace, dict):
        raise ValueError(f"{log_path}: completed record is missing sid or trace")
    if trace.get("status") != "done":
        raise ValueError(f"{log_path}: trace status is not done")
    if not isinstance(trace.get("browser_total_seconds"), (int, float)):
        raise ValueError(f"{log_path}: browser_total_seconds is missing")
    trace = dict(trace)
    trace["benchmark_sid"] = sid
    trace["browser_wall_seconds"] = completed.get("wall_seconds")
    trace["benchmark_log"] = log_path.name
    return sid, trace


def import_logs(log_paths: list[Path], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    for log_path in log_paths:
        sid, trace = extract_trace(log_path)
        output_path = output_dir / f"{sid}.json"
        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{output_path}: existing trace is unreadable"
                ) from exc
            if existing.get("request_id") != trace.get("request_id"):
                raise FileExistsError(
                    f"{output_path}: sid {sid!r} already belongs to request "
                    f"{existing.get('request_id')!r}"
                )
            imported.append(
                {
                    "sid": sid,
                    "request_id": trace.get("request_id"),
                    "browser_total_seconds": existing.get(
                        "browser_total_seconds"
                    ),
                    "service_state": existing.get("service_state"),
                    "output": str(output_path),
                    "status": "existing",
                }
            )
            continue
        output_path.write_text(
            json.dumps(trace, indent=2) + "\n",
            encoding="utf-8",
        )
        imported.append(
            {
                "sid": sid,
                "request_id": trace.get("request_id"),
                "browser_total_seconds": trace["browser_total_seconds"],
                "service_state": trace.get("service_state"),
                "output": str(output_path),
                "status": "imported",
            }
        )
    return imported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/performance/runs"),
    )
    args = parser.parse_args()
    imported = import_logs(args.logs, args.output_dir)
    print(json.dumps({"imported": imported}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
