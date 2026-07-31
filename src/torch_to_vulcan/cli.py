"""Command-line entry points for compiler contract tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .importer import InspectionError, InspectionReport, inspect_path
from .importer.report import display_domain
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


def _inspect(
    path: Path,
    *,
    json_output: bool,
    summary_only: bool,
    force_large_model: bool,
) -> int:
    try:
        report = inspect_path(path, confirm_large_model=force_large_model)
    except InspectionError as error:
        print(f"error: {error}")
        return 1

    if json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_inspection_report(report, summary_only=summary_only)
    return 2 if report.errors else 0


def _print_inspection_report(report: InspectionReport, *, summary_only: bool) -> None:
    print(f"Source: {report.source}")
    print(f"ONNX models: {len(report.models)}")
    print(f"Operators: {report.operator_count}")

    print("\nOperator summary:")
    if report.operator_summary:
        for item in report.operator_summary:
            print(f"  {display_domain(item.domain)}::{item.op_type}: {item.count}")
    else:
        print("  (none)")

    if not summary_only:
        print("\nModels:")
        for model in report.models:
            print(f"  {model.path}")
            opsets = ", ".join(
                f"{display_domain(opset.domain)}={opset.version}" for opset in model.opsets
            )
            print(f"    graph: {model.graph_name}")
            print(f"    IR version: {model.ir_version}")
            print(f"    opsets: {opsets or '(none)'}")
            for graph in model.graphs:
                print(f"    graph {graph.path} ({len(graph.operators)} operators)")
                for operator in graph.operators:
                    name = f" name={operator.name!r}" if operator.name else ""
                    print(
                        f"      [{operator.index}] "
                        f"{display_domain(operator.domain)}::{operator.op_type}{name}"
                    )

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  {error.path}: {error.message}")


def _serve(host: str, port: int, reload: bool) -> int:
    try:
        import uvicorn
    except ImportError:
        print('error: Web dependencies are missing; install with pip install -e ".[web]"')
        return 1

    uvicorn.run("torch_to_vulcan.api:app", host=host, port=port, reload=reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttv")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a Graph IR JSON file")
    validate_parser.add_argument("path", type=Path)
    inspect_parser = subparsers.add_parser(
        "inspect", help="list operators in an ONNX model or supported archive"
    )
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--json", action="store_true", help="write the report as JSON")
    inspect_parser.add_argument(
        "--summary-only", action="store_true", help="omit the per-model operator listing"
    )
    inspect_parser.add_argument(
        "--force-large-model",
        action="store_true",
        help="continue when the estimated model size exceeds 60% of available memory",
    )
    serve_parser = subparsers.add_parser("serve", help="start the inspection HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    serve_parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args.path)
    if args.command == "inspect":
        return _inspect(
            args.path,
            json_output=args.json,
            summary_only=args.summary_only,
            force_large_model=args.force_large_model,
        )
    if args.command == "serve":
        return _serve(args.host, args.port, args.reload)
    raise GraphValidationError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
