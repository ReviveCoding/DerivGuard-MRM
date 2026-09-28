"""Windows-friendly DerivGuard command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from derivguard.research import (
    capture_environment,
    generate_reports,
    run_data_eda,
    run_full,
    run_numerical_benchmarks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m derivguard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("data")
    subparsers.add_parser("audit-data")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--device", default="cuda", choices=("cuda", "auto"))
    run = subparsers.add_parser("run")
    run.add_argument("--profile", default="research", choices=("research", "audit", "smoke"))
    run.add_argument("--device", default="cuda", choices=("cuda", "auto"))
    run.add_argument("--resume", action="store_true")
    subparsers.add_parser("report")
    full = subparsers.add_parser("full")
    full.add_argument("--profile", default="research", choices=("research", "audit", "smoke"))
    full.add_argument("--device", default="cuda", choices=("cuda", "auto"))
    full.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "device", "cuda") == "auto":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is unavailable; large research workloads will not silently use CPU"
            )
    if args.command == "preflight":
        payload = capture_environment()
    elif args.command == "data":
        from derivguard.pipeline import run_data_stage

        payload = run_data_stage(Path.cwd(), resume=True)
    elif args.command == "audit-data":
        payload = run_data_eda()
    elif args.command == "smoke":
        capture_environment()
        payload = run_numerical_benchmarks("smoke")
    elif args.command == "run":
        payload = run_full(args.profile, device=args.device, resume=args.resume)
    elif args.command == "report":
        generate_reports()
        payload = {"status": "REPORTS_GENERATED"}
    elif args.command == "full":
        payload = run_full(args.profile, device=args.device, resume=args.resume)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {args.command}")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
