#!/usr/bin/env python3
"""Update experiment-index.json from a run summary.json."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append or update experiment-index.json from a run summary.json."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Training output directory containing experiment-index.json.",
    )
    parser.add_argument(
        "--summary-path",
        required=True,
        help="Path to a run summary.json file.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def normalize_run_entry(summary: Dict[str, Any], summary_relpath: str) -> Dict[str, Any]:
    metrics = summary.get("metrics") or {}
    metrics_out: Dict[str, Any] = {}
    for key in ("accuracy", "loss", "top_k"):
        if key in metrics and metrics[key] is not None:
            metrics_out[key] = metrics[key]

    return {
        "run_id": summary.get("run_id"),
        "display_name": summary.get("display_name") or summary.get("run_id"),
        "status": summary.get("status") or "completed",
        "category": summary.get("category") or "uncategorized",
        "dataset": summary.get("dataset"),
        "metrics": metrics_out,
        "summary_path": summary_relpath,
    }


def update_index(index_path: str, run_entry: Dict[str, Any]) -> Dict[str, Any]:
    if os.path.exists(index_path):
        index_data = load_json(index_path)
    else:
        index_data = {"schema_version": "1.0", "generated_at": None, "runs": []}

    runs = index_data.get("runs")
    if not isinstance(runs, list):
        runs = []

    run_id = run_entry.get("run_id")
    if not run_id:
        raise ValueError("summary.json must include run_id")

    updated_runs = [entry for entry in runs if entry.get("run_id") != run_id]
    updated_runs.append(run_entry)

    index_data["schema_version"] = index_data.get("schema_version") or "1.0"
    index_data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index_data["runs"] = sorted(
        updated_runs, key=lambda entry: entry.get("run_id") or ""
    )

    return index_data


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    summary_path = os.path.abspath(args.summary_path)
    index_path = os.path.join(output_dir, "experiment-index.json")

    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("summary.json must be a single JSON object")

    summary_relpath = os.path.relpath(summary_path, output_dir)
    if summary_relpath.startswith(".."):
        summary_relpath = summary_path

    run_entry = normalize_run_entry(summary, summary_relpath)
    index_data = update_index(index_path, run_entry)
    write_json(index_path, index_data)

    print(f"Updated {index_path} with run {run_entry['run_id']}")


if __name__ == "__main__":
    main()
