"""Command-line entry points for compiler contract tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .ir import GraphValidationError, graph_from_dict


def _validate(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        graph = graph_from_dict(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 1

    print(
        f"valid Graph IR {graph.schema_version}: {graph.model.name} "
        f"({len(graph.nodes)} nodes, {len(graph.tensors)} tensors)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttv")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a Graph IR JSON file")
    validate_parser.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args.path)
    raise GraphValidationError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

