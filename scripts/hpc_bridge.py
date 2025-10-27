#!/usr/bin/env python3
"""
HPC Bridge CLI (Phase 2 scaffold)

Prepares simple job folders for persona vector extraction, dataset screening, or
steering runs on an HPC environment, and prints rsync commands to sync
artifacts between local and remote.

This is a lightweight, cluster-agnostic scaffold. Customize SLURM/cluster
templates as needed for your environment.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def prepare_extraction(args):
    job_dir = Path(args.out_dir) / f"extract_{int(datetime.utcnow().timestamp())}"
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "extraction",
        "model": args.model,
        "traits": args.traits,
        "layer": args.layer,
        "positive_prompts": args.positive_prompts,
        "negative_prompts": args.negative_prompts,
        "output_vectors_dir": "vectors/",
    }
    _write_json(job_dir / "config.json", config)

    print(f"Prepared extraction job at: {job_dir}")
    print("Suggested rsync push:")
    print(f"  rsync -avz {job_dir}/ $USER@<hpc-host>:/path/to/jobs/{job_dir.name}/")


def prepare_screening(args):
    job_dir = Path(args.out_dir) / f"screen_{int(datetime.utcnow().timestamp())}"
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "screening",
        "model": args.model,
        "dataset_path": args.dataset,
        "text_column": args.text_column,
        "traits": args.traits,
        "vectors_dir": args.vectors_dir,
        "report_out": "reports/report.json",
    }
    _write_json(job_dir / "config.json", config)

    print(f"Prepared screening job at: {job_dir}")
    print("Suggested rsync push:")
    print(f"  rsync -avz {job_dir}/ $USER@<hpc-host>:/path/to/jobs/{job_dir.name}/")


def prepare_steering(args):
    job_dir = Path(args.out_dir) / f"steer_{int(datetime.utcnow().timestamp())}"
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "steering",
        "model": args.model,
        "prompt": args.prompt,
        "trait": args.trait,
        "multiplier": args.multiplier,
        "layer": args.layer,
        "vectors_dir": args.vectors_dir,
        "runs_out": "runs/",
    }
    _write_json(job_dir / "config.json", config)

    print(f"Prepared steering job at: {job_dir}")
    print("Suggested rsync push:")
    print(f"  rsync -avz {job_dir}/ $USER@<hpc-host>:/path/to/jobs/{job_dir.name}/")


def print_sync(args):
    local = Path(args.local).resolve()
    remote = args.remote.rstrip("/")
    direction = args.direction

    if direction == "push":
        print(f"rsync -avz {local}/ {remote}/")
    else:
        print(f"rsync -avz {remote}/ {local}/")


def main():
    parser = argparse.ArgumentParser(description="HPC Bridge CLI (scaffold)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Extraction
    p_ext = sub.add_parser("prepare-extraction", help="Prepare extraction job folder")
    p_ext.add_argument("--model", default="google/gemma-3-4b")
    p_ext.add_argument("--traits", nargs='+', required=True)
    p_ext.add_argument("--layer", type=int, default=-1)
    p_ext.add_argument("--positive-prompts", nargs='*', default=[])
    p_ext.add_argument("--negative-prompts", nargs='*', default=[])
    p_ext.add_argument("--out-dir", default="jobs")
    p_ext.set_defaults(func=prepare_extraction)

    # Screening
    p_scr = sub.add_parser("prepare-screening", help="Prepare screening job folder")
    p_scr.add_argument("--model", default="google/gemma-3-4b")
    p_scr.add_argument("--dataset", required=True, help="Path to .jsonl dataset")
    p_scr.add_argument("--text-column", default="text")
    p_scr.add_argument("--traits", nargs='+', required=True)
    p_scr.add_argument("--vectors-dir", default="vectors")
    p_scr.add_argument("--out-dir", default="jobs")
    p_scr.set_defaults(func=prepare_screening)

    # Steering
    p_str = sub.add_parser("prepare-steering", help="Prepare steering job folder")
    p_str.add_argument("--model", default="google/gemma-3-4b")
    p_str.add_argument("--prompt", required=True)
    p_str.add_argument("--trait", required=True)
    p_str.add_argument("--multiplier", type=float, default=1.5)
    p_str.add_argument("--layer", type=int, default=20)
    p_str.add_argument("--vectors-dir", default="vectors")
    p_str.add_argument("--out-dir", default="jobs")
    p_str.set_defaults(func=prepare_steering)

    # Sync helper
    p_sync = sub.add_parser("sync", help="Print rsync command for push/pull")
    p_sync.add_argument("direction", choices=["push", "pull"])  # local -> remote or remote -> local
    p_sync.add_argument("--local", required=True)
    p_sync.add_argument("--remote", required=True, help="$USER@host:/remote/path")
    p_sync.set_defaults(func=print_sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


