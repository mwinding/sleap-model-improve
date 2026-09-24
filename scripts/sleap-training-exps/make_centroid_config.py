#!/usr/bin/env python3
"""Create a centroid training config from real data plus chosen synthetic SLPs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("scripts/sleap-training-exps/configs/centroid_holdout.yaml"),
                        help="Base config; the default trains on real labels without held-out benchmark frames.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--synthetic", type=Path, nargs="+", required=True,
                        help="One or more SLP files to add after the real labels.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    synthetic_paths = [path.resolve() for path in args.synthetic]
    missing = [str(path) for path in synthetic_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Synthetic label files not found: {', '.join(missing)}")

    config["data_config"]["train_labels_path"] = [
        *config["data_config"]["train_labels_path"],
        *(str(path) for path in synthetic_paths),
    ]
    config["trainer_config"]["run_name"] = args.run_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    print("Training label files:")
    for path in config["data_config"]["train_labels_path"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()