"""Command-line entry points for compiler contract tooling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from .compiler import (
    ContractError,
    ExecutablePackageError,
    ShapeProfile,
    StaticCompilationError,
    compile_static_onnx,
    validate_executable_package,
)
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


def _validate_package(path: Path) -> int:
    try:
        manifest = validate_executable_package(path)
    except ExecutablePackageError as error:
        print(f"error: {error}")
        return 1
    programs = manifest["programs"]
    dispatch_count = sum(len(program["steps"]) for program in programs)
    print(
        f"valid TTV executable package {manifest['format_version']}: "
        f"{manifest['model_name']} ({dispatch_count} dispatches, "
        f"{len(manifest['shaders'])} shaders)"
    )
    return 0


def _load_shape_profile(path: Path | None) -> ShapeProfile | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ContractError("shape profile JSON must be an object")
        return ShapeProfile.from_dict(value)
    except (OSError, json.JSONDecodeError, ContractError, TypeError, ValueError) as error:
        raise ExecutablePackageError(f"cannot read shape profile: {error}") from error


def _compile(
    source: Path,
    destination: Path,
    *,
    include_debug_sources: bool,
    shape_profile_path: Path | None = None,
) -> int:
    try:
        shape_profile = _load_shape_profile(shape_profile_path)
        report = compile_static_onnx(
            source,
            destination,
            include_debug_sources=include_debug_sources,
            shape_profile=shape_profile,
        )
    except StaticCompilationError as error:
        for diagnostic in error.diagnostics:
            print(diagnostic)
        return 1
    except ExecutablePackageError as error:
        print(f"error: {error}")
        return 1
    print(
        f"compiled TTV executable package {report.manifest['format_version']}: "
        f"{report.manifest['model_name']} ({report.folded_nodes} folded nodes, "
        f"{report.dispatches} dispatches, {report.metadata_views} views)"
    )
    for diagnostic in report.diagnostics:
        print(diagnostic)
    print(f"output: {report.destination}")
    return 0


def _load_npz_inputs(path: Path) -> dict[str, object]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise ExecutablePackageError(f"cannot read input .npz: {error}") from error


def _run_package(
    package: Path,
    input_npz: Path,
    output_npz: Path | None,
    *,
    device_local: bool = False,
) -> int:
    try:
        from .compiler.vulkan.runtime import VulkanPackageRuntime

        inputs = _load_npz_inputs(input_npz)
        with VulkanPackageRuntime(package, device_local=device_local) as runtime:
            result = runtime.run(inputs)
            memory_mode = runtime.memory_mode
        if output_npz is not None:
            import numpy as np

            np.savez(output_npz, **result.outputs)
        requested_mode = "device-local" if device_local else "host-visible"
        print(
            f"executed {package}: {result.device_name}, {result.elapsed_ms:.3f} ms "
            f"(memory={memory_mode}, requested={requested_mode})"
        )
        for tensor_id, value in result.outputs.items():
            print(f"  {tensor_id}: dtype={value.dtype}, shape={value.shape}")
        if output_npz is not None:
            print(f"outputs: {output_npz}")
        return 0
    except (ExecutablePackageError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1


def _benchmark_package(
    package: Path,
    input_npz: Path,
    *,
    warmup: int,
    iterations: int,
    cpu_onnx: Path | None,
    resident: bool,
    device_local: bool = False,
) -> int:
    try:
        from .compiler.vulkan.runtime import VulkanPackageRuntime

        inputs = _load_npz_inputs(input_npz)
        with VulkanPackageRuntime(package, device_local=device_local) as runtime:
            result = runtime.benchmark(
                inputs,
                warmup=warmup,
                iterations=iterations,
                resident=resident,
            )
            memory_mode = runtime.memory_mode
        requested_mode = "device-local" if device_local else "host-visible"
        print(
            f"benchmarked {package}: {result.device_name}, "
            f"mean={result.mean_ms:.3f} ms, median={result.median_ms:.3f} ms, "
            f"min={result.min_ms:.3f} ms, max={result.max_ms:.3f} ms "
            f"({result.iterations} iterations, {result.warmup} warmup, {result.scope}, "
            f"memory={memory_mode}, requested={requested_mode})"
        )
        print(
            f"  stages: upload={result.mean_upload_ms:.3f} ms, "
            f"dispatch+wait={result.mean_dispatch_ms:.3f} ms, "
            f"readback={result.mean_readback_ms:.3f} ms"
        )
        if cpu_onnx is not None:
            cpu_mean = _benchmark_cpu_onnx(
                cpu_onnx,
                inputs,
                warmup=warmup,
                iterations=iterations,
            )
            print(f"CPU ONNX Runtime: mean={cpu_mean:.3f} ms")
            if result.mean_ms > 0:
                print(f"GPU speedup vs CPU: {cpu_mean / result.mean_ms:.2f}x")
        return 0
    except (ExecutablePackageError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1


def _benchmark_cpu_onnx(
    source: Path,
    inputs: Mapping[str, object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ExecutablePackageError(
            "CPU 对比需要 onnxruntime，请安装 verify 依赖"
        ) from error
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    feeds = {name: value for name, value in inputs.items() if name in input_names}
    missing = input_names - set(feeds)
    if missing:
        raise ExecutablePackageError(f"CPU 对比缺少输入: {', '.join(sorted(missing))}")
    for _ in range(warmup):
        session.run(None, feeds)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, feeds)
        timings.append((time.perf_counter() - started) * 1000.0)
    return float(sum(timings) / len(timings))


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
    package_parser = subparsers.add_parser(
        "validate-package",
        help="validate a materialized TTV executable directory",
    )
    package_parser.add_argument("path", type=Path)
    compile_parser = subparsers.add_parser(
        "compile",
        help="compile a static linear ONNX root graph to a TTV package",
    )
    compile_parser.add_argument("source", type=Path)
    compile_parser.add_argument("destination", type=Path)
    compile_parser.add_argument(
        "--no-debug-sources",
        action="store_true",
        help="omit generated GLSL source files from the package",
    )
    compile_parser.add_argument(
        "--shape-profile",
        type=Path,
        help="JSON profile binding symbolic ONNX dimensions for this compilation",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="execute a validated TTV package with inputs from an .npz file",
    )
    run_parser.add_argument("package", type=Path)
    run_parser.add_argument("input_npz", type=Path)
    run_parser.add_argument("--output", dest="output_npz", type=Path)
    run_parser.add_argument(
        "--device-local",
        action="store_true",
        help="prefer device-local tensor buffers with staging transfers",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark a TTV package with persistent Vulkan resources",
    )
    benchmark_parser.add_argument("package", type=Path)
    benchmark_parser.add_argument("input_npz", type=Path)
    benchmark_parser.add_argument("--warmup", type=int, default=3)
    benchmark_parser.add_argument("--iterations", type=int, default=10)
    benchmark_parser.add_argument(
        "--cpu-onnx",
        type=Path,
        help="also benchmark ONNX Runtime CPU with this source model",
    )
    benchmark_parser.add_argument(
        "--resident",
        action="store_true",
        help="benchmark dispatch only with inputs and outputs resident on Vulkan buffers",
    )
    benchmark_parser.add_argument(
        "--device-local",
        action="store_true",
        help="prefer device-local tensor buffers with staging transfers",
    )
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
    if args.command == "validate-package":
        return _validate_package(args.path)
    if args.command == "compile":
        return _compile(
            args.source,
            args.destination,
            include_debug_sources=not args.no_debug_sources,
            shape_profile_path=args.shape_profile,
        )
    if args.command == "run":
        return _run_package(
            args.package,
            args.input_npz,
            args.output_npz,
            device_local=args.device_local,
        )
    if args.command == "benchmark":
        return _benchmark_package(
            args.package,
            args.input_npz,
            warmup=args.warmup,
            iterations=args.iterations,
            cpu_onnx=args.cpu_onnx,
            resident=args.resident,
            device_local=args.device_local,
        )
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
