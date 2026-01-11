#!/usr/bin/env python3
"""Merge multiple experiment-index.json files into a single UI-friendly index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple experiment-index.json files into one.",
    )
    parser.add_argument(
        "--index",
        action="append",
        required=True,
        help="Path to an experiment-index.json file (repeatable).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the combined index file.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory used to build absolute summary_path URLs.",
    )
    return parser.parse_args()


def load_index(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Index file is not a JSON object: {path}")
    return data


def resolve_summary_path(index_path: Path, root_dir: Path, summary_path: str) -> str:
    prefix = "/" + str(index_path.parent.relative_to(root_dir)).replace("\\", "/")
    summary_path = summary_path.lstrip("/")
    return f"{prefix}/{summary_path}"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root).resolve()

    combined_runs: dict[str, dict] = {}
    for index_value in args.index:
        index_path = Path(index_value).resolve()
        data = load_index(index_path)
        runs = data.get("runs", [])
        if not isinstance(runs, list):
            raise ValueError(f"Index runs must be a list: {index_path}")

        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"Run entry must be an object: {index_path}")
            run_id = run.get("run_id")
            if not run_id:
                continue
            run_entry = dict(run)
            summary_path = run_entry.get("summary_path")
            if summary_path:
                run_entry["summary_path"] = resolve_summary_path(index_path, root_dir, summary_path)
            combined_runs[run_id] = run_entry

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": sorted(combined_runs.values(), key=lambda item: item.get("run_id") or ""),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(output_path)


if __name__ == "__main__":
    main()
