"""Static root-graph ONNX to TTV executable package compilation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import onnx
import numpy as np
from onnx import ModelProto, shape_inference

from .constants import fold_constant_subgraph, rewrite_model_with_folded_constants
from .contracts import ShapeProfile
from .onnx.ir import (
    NormalizedConstant,
    NormalizedNode,
    NormalizedTensor,
    NormalizationError,
    normalize_onnx_model,
)
from .package import DATA_TYPE_BYTES, ExecutablePackageBuilder, ExecutablePackageError
from .vulkan.ir import DispatchPlan
from .vulkan.kernels import (
    KernelContext,
    KernelRegistry,
    KernelTensor,
    UnsupportedKernel,
    default_kernel_registry,
)
from .vulkan.verify import VerificationRunner

ShaderCompiler = Callable[[str], bytes]

CONTROL_FLOW_OPERATORS = frozenset({"If", "Loop", "Scan"})


@dataclass(frozen=True, slots=True)
class CompilationDiagnostic:
    severity: str
    code: str
    message: str
    node_index: int | None = None
    node_name: str = ""
    op_type: str = ""

    def __str__(self) -> str:
        location = "model"
        if self.node_index is not None:
            label = self.node_name or self.op_type or "node"
            location = f"node[{self.node_index}] {label}"
        return f"{self.severity} {self.code} at {location}: {self.message}"


class StaticCompilationError(ExecutablePackageError):
    """The model cannot be represented by the current static package format."""

    def __init__(self, diagnostics: tuple[CompilationDiagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        errors = [item for item in diagnostics if item.severity == "error"]
        summary = f"static compilation failed with {len(errors)} error(s)"
        if errors:
            summary += f": {errors[0]}"
        super().__init__(summary)


@dataclass(frozen=True, slots=True)
class CompilationReport:
    destination: Path
    manifest: dict[str, object]
    folded_nodes: int
    runtime_nodes: int
    dispatches: int
    metadata_views: int
    diagnostics: tuple[CompilationDiagnostic, ...]
    shape_profile: ShapeProfile | None = None


def compile_static_model(
    model: ModelProto,
    destination: str | Path,
    *,
    model_name: str | None = None,
    registry: KernelRegistry | None = None,
    shader_compiler: ShaderCompiler | None = None,
    include_debug_sources: bool = True,
    metadata: Mapping[str, str] | None = None,
    shape_profile: ShapeProfile | None = None,
) -> CompilationReport:
    """Compile one linear ONNX root graph, optionally specialized by a profile."""
    diagnostics: list[CompilationDiagnostic] = []
    try:
        onnx.checker.check_model(model)
    except Exception as error:
        raise StaticCompilationError(
            (CompilationDiagnostic("error", "INVALID_ONNX", str(error)),)
        ) from error

    folded = fold_constant_subgraph(model)
    diagnostics.extend(
        CompilationDiagnostic(
            "warning",
            "CONSTANT_FOLD_SKIPPED",
            item.message,
            None if item.index < 0 else item.index,
            item.name,
            item.op_type,
        )
        for item in folded.diagnostics
    )
    rewritten = rewrite_model_with_folded_constants(model, folded)
    try:
        onnx.checker.check_model(rewritten)
        inferred = shape_inference.infer_shapes(
            rewritten,
            check_type=True,
            strict_mode=True,
            data_prop=True,
        )
    except Exception as error:
        raise StaticCompilationError(
            tuple(diagnostics)
            + (CompilationDiagnostic("error", "SHAPE_INFERENCE_FAILED", str(error)),)
        ) from error

    try:
        normalized = normalize_onnx_model(inferred)
    except NormalizationError as error:
        raise StaticCompilationError(
            tuple(diagnostics)
            + (CompilationDiagnostic("error", "NORMALIZATION_FAILED", str(error)),)
        ) from error
    if shape_profile is not None:
        try:
            normalized = normalized.specialize(shape_profile)
        except NormalizationError as error:
            raise StaticCompilationError(
                tuple(diagnostics)
                + (
                    CompilationDiagnostic(
                        "error",
                        "PROFILE_DIMENSION_UNBOUND",
                        str(error),
                    ),
                )
            ) from error

    graph = normalized.graph
    specs = graph.tensors
    _diagnose_static_shapes(specs, shape_profile, diagnostics)
    selected_registry = registry or default_kernel_registry()
    compile_shader = shader_compiler or _compile_shader
    package_name = model_name or graph.name or "onnx-model"
    builder = ExecutablePackageBuilder(package_name)
    if shape_profile is not None:
        builder.add_profile(shape_profile.name, shape_profile.dimensions)
    initializer_names = set(graph.constants)
    graph_outputs = set(graph.outputs)

    for input_name in graph.inputs:
        if input_name in initializer_names:
            continue
        spec = specs.get(input_name)
        if not _require_static_spec(spec, input_name, diagnostics):
            continue
        try:
            builder.add_tensor(input_name, spec.data_type, spec.static_shape, storage="external")
            builder.bind_input(input_name)
        except ExecutablePackageError as error:
            diagnostics.append(CompilationDiagnostic("error", "INPUT_TENSOR", str(error)))

    referenced_values = {
        name for node in graph.nodes for name in node.inputs if name
    } | graph_outputs
    for initializer_name, constant in graph.constants.items():
        if initializer_name not in referenced_values:
            continue
        try:
            _add_initializer(builder, constant)
        except (ExecutablePackageError, OSError, TypeError, ValueError) as error:
            diagnostics.append(
                CompilationDiagnostic(
                    "error",
                    "CONSTANT_MATERIALIZATION",
                    f"{initializer_name}: {error}",
                )
            )

    dispatch_count = 0
    metadata_views = 0
    shader_cache: dict[str, bytes] = {}
    for node_index, node in enumerate(graph.nodes):
        node_diagnostics: list[CompilationDiagnostic] = []
        if node.op_type in CONTROL_FLOW_OPERATORS or node.has_subgraphs:
            diagnostics.append(
                _node_diagnostic(
                    node_index,
                    node,
                    "CONTROL_FLOW_UNSUPPORTED",
                    "TTV package 0.1 only supports the root linear program",
                )
            )
            continue

        input_names = tuple(name for name in node.inputs if name)
        output_names = node.outputs
        input_specs = _node_specs(input_names, specs, node_index, node, node_diagnostics)
        output_specs = _node_specs(output_names, specs, node_index, node, node_diagnostics)
        for name in input_names:
            if name not in builder.tensors:
                node_diagnostics.append(
                    _node_diagnostic(
                        node_index,
                        node,
                        "INPUT_NOT_MATERIALIZED",
                        f"input tensor {name!r} has no earlier definition",
                    )
                )
        diagnostics.extend(node_diagnostics)
        if node_diagnostics:
            continue

        context = KernelContext(
            node.domain,
            node.op_type,
            node.opset_version,
            dict(node.attributes),
            tuple(
                _kernel_tensor(name, spec)
                for name, spec in zip(input_names, input_specs, strict=True)
            ),
            tuple(
                _kernel_tensor(name, spec)
                for name, spec in zip(output_names, output_specs, strict=True)
            ),
            constant_inputs=_compile_time_constants(input_names, graph.constants),
        )
        try:
            plan = selected_registry.select(context)
        except UnsupportedKernel as error:
            diagnostics.append(
                _node_diagnostic(node_index, node, "KERNEL_UNSUPPORTED", str(error))
            )
            continue

        try:
            if plan.metadata_only:
                _materialize_view(builder, plan, input_names, output_names, output_specs)
                metadata_views += 1
                continue

            for name, spec in zip(output_names, output_specs, strict=True):
                builder.add_tensor(
                    name,
                    spec.data_type,
                    spec.static_shape,
                    storage="external" if name in graph_outputs else "transient",
                )
            for step_index, step in enumerate(plan.steps):
                source = step.shader.source
                if source not in shader_cache:
                    shader_cache[source] = compile_shader(source)
                node_id = _dispatch_id(node_index, node.name or node.op_type, step_index, plan)
                builder.add_dispatch(
                    node_id,
                    plan.kernel_id,
                    step,
                    shader_cache[source],
                    tuple(binding.name for binding in step.bindings),
                )
                dispatch_count += 1
        except (ExecutablePackageError, OSError, RuntimeError, ValueError) as error:
            diagnostics.append(
                _node_diagnostic(node_index, node, "LOWERING_FAILED", str(error))
            )

    for output_name in graph.outputs:
        if output_name not in builder.tensors:
            diagnostics.append(
                CompilationDiagnostic(
                    "error",
                    "OUTPUT_NOT_MATERIALIZED",
                    f"output tensor {output_name!r} has no compiled definition",
                )
            )
            continue
        try:
            builder.bind_output(output_name)
        except ExecutablePackageError as error:
            diagnostics.append(CompilationDiagnostic("error", "OUTPUT_TENSOR", str(error)))

    if any(item.severity == "error" for item in diagnostics):
        raise StaticCompilationError(tuple(diagnostics))

    package_metadata = {"compiler": "torch-to-vulcan-static-0.1"}
    if shape_profile is not None:
        package_metadata["shape_profile"] = json.dumps(
            shape_profile.to_dict(), ensure_ascii=True, sort_keys=True
        )
    if metadata:
        package_metadata.update(metadata)
    destination_path = Path(destination)
    manifest = builder.write(
        destination_path,
        include_debug_sources=include_debug_sources,
        metadata=package_metadata,
    )
    return CompilationReport(
        destination_path,
        manifest,
        len(folded.folded_nodes),
        len(graph.nodes),
        dispatch_count,
        metadata_views,
        tuple(diagnostics),
        shape_profile,
    )


def compile_static_onnx(
    source: str | Path,
    destination: str | Path,
    *,
    model_name: str | None = None,
    registry: KernelRegistry | None = None,
    shader_compiler: ShaderCompiler | None = None,
    include_debug_sources: bool = True,
    metadata: Mapping[str, str] | None = None,
    shape_profile: ShapeProfile | None = None,
) -> CompilationReport:
    """Load an ONNX file, including external data, and compile its root graph."""
    source_path = Path(source)
    try:
        model = onnx.load_model(source_path, load_external_data=True)
    except (OSError, ValueError) as error:
        raise StaticCompilationError(
            (CompilationDiagnostic("error", "ONNX_LOAD_FAILED", str(error)),)
        ) from error
    package_metadata = dict(metadata or {})
    package_metadata.setdefault("source", str(source_path))
    return compile_static_model(
        model,
        destination,
        model_name=model_name,
        registry=registry,
        shader_compiler=shader_compiler,
        include_debug_sources=include_debug_sources,
        metadata=package_metadata,
        shape_profile=shape_profile,
    )


def _compile_shader(source: str) -> bytes:
    spirv, _stage, message = VerificationRunner().compile_shader(source)
    if spirv is None:
        raise ExecutablePackageError(message)
    return spirv


def _diagnose_static_shapes(
    specs: Mapping[str, NormalizedTensor],
    profile: ShapeProfile | None,
    diagnostics: list[CompilationDiagnostic],
) -> None:
    """Keep profile and unknown-shape diagnostics at the compiler boundary."""
    for tensor_name, spec in specs.items():
        if spec.shape is None:
            diagnostics.append(
                CompilationDiagnostic(
                    "error",
                    "DYNAMIC_SHAPE_UNSUPPORTED",
                    f"tensor {tensor_name!r} has no known rank",
                )
            )
        elif any(dimension is None or isinstance(dimension, str) for dimension in spec.shape):
            code = (
                "PROFILE_DIMENSION_UNBOUND"
                if profile is not None
                else "DYNAMIC_SHAPE_UNSUPPORTED"
            )
            message = (
                f"tensor {tensor_name!r} remains symbolic after shape specialization"
                if profile is not None
                else f"tensor {tensor_name!r} has symbolic dimensions; provide a shape profile"
            )
            diagnostics.append(CompilationDiagnostic("error", code, message))


def _require_static_spec(
    spec: NormalizedTensor | None,
    tensor_name: str,
    diagnostics: list[CompilationDiagnostic],
) -> bool:
    if spec is None:
        diagnostics.append(
            CompilationDiagnostic(
                "error", "TENSOR_TYPE_MISSING", f"tensor {tensor_name!r} has no type record"
            )
        )
        return False
    if not spec.fully_static:
        diagnostics.append(
            CompilationDiagnostic(
                "error",
                "DYNAMIC_SHAPE_UNSUPPORTED",
                f"tensor {tensor_name!r} does not have a fully static shape",
            )
        )
        return False
    if spec.data_type in {"UNDEFINED", "UNKNOWN"}:
        diagnostics.append(
            CompilationDiagnostic(
                "error",
                "TENSOR_TYPE_MISSING",
                f"tensor {tensor_name!r} has data type {spec.data_type}",
            )
        )
        return False
    return True


def _node_specs(
    names: tuple[str, ...],
    specs: Mapping[str, NormalizedTensor],
    node_index: int,
    node: NormalizedNode,
    diagnostics: list[CompilationDiagnostic],
) -> tuple[NormalizedTensor, ...]:
    result: list[NormalizedTensor] = []
    for name in names:
        spec = specs.get(name)
        local: list[CompilationDiagnostic] = []
        if not _require_static_spec(spec, name, local):
            diagnostics.extend(
                _node_diagnostic(node_index, node, item.code, item.message)
                for item in local
            )
            continue
        result.append(spec)
    return tuple(result)


def _add_initializer(builder: ExecutablePackageBuilder, constant: NormalizedConstant) -> None:
    tensor = constant.tensor
    data_type = tensor.data_type
    expected_bytes = DATA_TYPE_BYTES.get(data_type)
    if expected_bytes is None:
        raise ExecutablePackageError(f"unsupported initializer data type {data_type}")
    expected_length = tensor.element_count * expected_bytes
    if len(constant.data) != expected_length:
        raise ExecutablePackageError(
            f"decoded {data_type} byte length {len(constant.data)} does not match {expected_length}"
        )
    builder.add_constant(tensor.name, data_type, tensor.static_shape, constant.data)


def _kernel_tensor(name: str, spec: NormalizedTensor) -> KernelTensor:
    if not spec.fully_static:
        return KernelTensor(name, spec.data_type, (), shape_known=False, layout=spec.layout)
    return KernelTensor(
        name,
        spec.data_type,
        tuple(str(value) for value in spec.static_shape),
        shape_known=True,
        layout=spec.layout,
        strides=tuple(str(value) for value in (spec.strides or ())),
    )


def _compile_time_constants(
    names: tuple[str, ...],
    constants: Mapping[str, NormalizedConstant],
) -> dict[str, object]:
    """Decode only small initializer controls used while selecting a kernel."""
    result: dict[str, object] = {}
    for name in names:
        constant = constants.get(name)
        if constant is None:
            continue
        tensor = constant.tensor
        if tensor.element_count > 16 or len(constant.data) > 128:
            continue
        dtype = _compile_time_numpy_dtype(tensor.data_type)
        if dtype is None:
            continue
        values = np.frombuffer(
            constant.data,
            dtype=dtype,
            count=tensor.element_count,
        ).copy()
        result[name] = values.reshape(tensor.static_shape)
    return result


def _compile_time_numpy_dtype(data_type: str) -> np.dtype[object] | None:
    dtypes: dict[str, np.dtype[object]] = {
        "BOOL": np.dtype(np.bool_),
        "FLOAT16": np.dtype(np.float16),
        "FLOAT": np.dtype(np.float32),
        "DOUBLE": np.dtype(np.float64),
        "INT8": np.dtype(np.int8),
        "UINT8": np.dtype(np.uint8),
        "INT16": np.dtype(np.int16),
        "UINT16": np.dtype(np.uint16),
        "INT32": np.dtype(np.int32),
        "UINT32": np.dtype(np.uint32),
        "INT64": np.dtype(np.int64),
        "UINT64": np.dtype(np.uint64),
    }
    return dtypes.get(data_type)


def _materialize_view(
    builder: ExecutablePackageBuilder,
    plan: DispatchPlan,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    output_specs: tuple[NormalizedTensor, ...],
) -> None:
    if len(output_names) != 1 or not input_names:
        raise ExecutablePackageError(
            f"metadata kernel {plan.kernel_id} requires one output and one source"
        )
    output_spec = output_specs[0]
    builder.add_view(
        output_names[0],
        output_spec.data_type,
        output_spec.static_shape,
        input_names[0],
        output_spec.strides or (),
    )


def _node_diagnostic(
    node_index: int,
    node: NormalizedNode,
    code: str,
    message: str,
) -> CompilationDiagnostic:
    return CompilationDiagnostic(
        "error", code, message, node_index, node.name, node.op_type
    )


def _dispatch_id(
    node_index: int,
    label: str,
    step_index: int,
    plan: DispatchPlan,
) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or plan.operator
    suffix = f"_step_{step_index}" if len(plan.steps) > 1 else ""
    return f"node_{node_index}_{safe_label}{suffix}"
