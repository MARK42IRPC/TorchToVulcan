"""Auditable staged verification for generated Vulkan mappings."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from onnx import TensorProto, helper

from .executor import VulkanExecutionError, VulkanExecutor
from .kernels import (
    KernelContext,
    KernelRegistry,
    KernelTensor,
    UnsupportedKernel,
    default_kernel_registry,
)

Publish = Callable[[dict[str, object]], None]

_FP32_UNARY_OPERATORS = {
    "Relu",
    "Neg",
    "Exp",
    "Floor",
    "Sin",
    "Cos",
    "Sqrt",
    "Tanh",
    "Sigmoid",
    "LeakyRelu",
}


@dataclass(frozen=True, slots=True)
class VerificationTensor:
    name: str
    data_type: str
    shape: tuple[str, ...]
    shape_known: bool = True


@dataclass(frozen=True, slots=True)
class VerificationTarget:
    target_id: str
    semantic_key: str
    domain: str
    op_type: str
    opset_version: int
    attributes: dict[str, object]
    inputs: tuple[VerificationTensor, ...]
    outputs: tuple[VerificationTensor, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VerificationTarget":
        def tensors(name: str) -> tuple[VerificationTensor, ...]:
            items = value.get(name, [])
            if not isinstance(items, list):
                raise ValueError(f"{name} must be an array")
            return tuple(
                VerificationTensor(
                    name=str(item.get("name", "")),
                    data_type=str(item.get("data_type", "UNKNOWN")),
                    shape=tuple(str(dimension) for dimension in item.get("shape", [])),
                    shape_known=bool(item.get("shape_known", True)),
                )
                for item in items
            )

        return cls(
            target_id=str(value.get("target_id", "")),
            semantic_key=str(value.get("semantic_key", "")),
            domain=str(value.get("domain", "")),
            op_type=str(value.get("op_type", "")),
            opset_version=int(value.get("opset_version", 0)),
            attributes=dict(value.get("attributes", {})),
            inputs=tensors("inputs"),
            outputs=tensors("outputs"),
        )


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    feeds: dict[str, np.ndarray[Any, Any]]
    outputs: tuple[np.ndarray[Any, Any], ...]
    digest: str
    summary: str


@dataclass(frozen=True, slots=True)
class ToolchainCapabilities:
    vulkaninfo: str | None
    glslang_validator: str | None
    spirv_val: str | None
    onnxruntime: bool
    vulkan_binding: bool
    executor_available: bool
    device_name: str


def detect_toolchain() -> ToolchainCapabilities:
    vulkaninfo = shutil.which("vulkaninfo")
    compiler = shutil.which("glslangValidator") or _portable_glslang()
    vulkan_binding = importlib.util.find_spec("vulkan") is not None
    return ToolchainCapabilities(
        vulkaninfo=vulkaninfo,
        glslang_validator=compiler,
        spirv_val=shutil.which("spirv-val"),
        onnxruntime=importlib.util.find_spec("onnxruntime") is not None,
        vulkan_binding=vulkan_binding,
        executor_available=vulkan_binding,
        device_name=_device_name(vulkaninfo),
    )


class VerificationRunner:
    def __init__(
        self,
        registry: KernelRegistry | None = None,
        capabilities: ToolchainCapabilities | None = None,
    ) -> None:
        self.registry = registry or default_kernel_registry()
        self.capabilities = capabilities or detect_toolchain()

    def run(self, targets: Iterable[VerificationTarget], publish: Publish) -> dict[str, int]:
        items = tuple(targets)
        total = len(items)
        publish(
            {
                "type": "started",
                "total": total,
                "capabilities": asdict(self.capabilities),
            }
        )
        summary = {"total": total, "verified": 0, "blocked": 0, "failed": 0}
        executor: VulkanExecutor | None = None
        try:
            if self.capabilities.executor_available:
                try:
                    executor = VulkanExecutor()
                    self._log(
                        publish,
                        "INFO",
                        "VULKAN",
                        f"Vulkan 执行器已连接: {executor.device_name}",
                    )
                except VulkanExecutionError as error:
                    self._log(publish, "WARN", "VULKAN", str(error))
            for current, target in enumerate(items, start=1):
                certificate = self._verify_target(
                    target,
                    current,
                    total,
                    publish,
                    executor,
                )
                status = str(certificate["status"])
                if status in {"RANGE_VERIFIED", "DEVICE_VERIFIED"}:
                    summary["verified"] += 1
                elif status == "FAILED":
                    summary["failed"] += 1
                else:
                    summary["blocked"] += 1
                publish({"type": "certificate", "certificate": certificate})
                publish(
                    {
                        "type": "progress",
                        "current": current,
                        "total": total,
                        "percent": round(current / total * 100, 2) if total else 100.0,
                        "operator": target.op_type,
                        "status": status,
                    }
                )
        finally:
            if executor is not None:
                executor.close()
        publish({"type": "result", "summary": summary})
        return summary

    def _verify_target(
        self,
        target: VerificationTarget,
        current: int,
        total: int,
        publish: Publish,
        executor: VulkanExecutor | None,
    ) -> dict[str, object]:
        label = f"{target.domain or 'ai.onnx'}::{target.op_type}"
        self._log(publish, "INFO", label, f"[{current}/{total}] 开始选择 Kernel Candidate")
        verification_target, normalization = _canonical_verification_target(target)
        if normalization:
            self._log(publish, "INFO", label, normalization)
        context = KernelContext(
            domain=verification_target.domain,
            op_type=verification_target.op_type,
            opset_version=verification_target.opset_version,
            attributes=verification_target.attributes,
            inputs=tuple(
                KernelTensor(item.name, item.data_type, item.shape, item.shape_known)
                for item in verification_target.inputs
            ),
            outputs=tuple(
                KernelTensor(item.name, item.data_type, item.shape, item.shape_known)
                for item in verification_target.outputs
            ),
        )
        try:
            plan = self.registry.select(context)
        except UnsupportedKernel as error:
            message = str(error)
            self._log(publish, "WARN", label, message)
            return self._certificate(target, "BLOCKED", "NONE", message)

        self._log(
            publish,
            "INFO",
            label,
            f"生成 {plan.kernel_id}，Dispatch 数量 {plan.dispatch_count}",
        )
        if plan.metadata_only:
            message = plan.notes[0] if plan.notes else "元数据执行计划"
            self._log(publish, "INFO", label, message)
            return self._certificate(
                target,
                "RANGE_VERIFIED",
                "PLAN_VERIFIED",
                message,
                plan.kernel_id,
                cases_total=1,
                cases_passed=1,
                metrics=[{"kind": "zero_dispatch_invariant"}],
                confidence=1.0,
            )

        shader = plan.steps[0].shader
        shader_hash = hashlib.sha256(shader.source.encode()).hexdigest()
        reference_hash = ""
        references: tuple[ReferenceCase, ...] = ()
        reference_stage = "GENERATED"
        if self.capabilities.onnxruntime:
            try:
                references, reference_hash, output_summary = _run_onnx_reference(
                    verification_target
                )
                reference_stage = "REFERENCE_EXECUTED"
                self._log(
                    publish,
                    "INFO",
                    label,
                    f"ONNX Runtime 参考输出完成: {output_summary} / {reference_hash[:12]}",
                )
            except Exception as error:
                message = f"ONNX Runtime 参考执行失败: {error}"
                self._log(publish, "ERROR", label, message)
                return self._certificate(
                    target,
                    "FAILED",
                    "GENERATED",
                    message,
                    plan.kernel_id,
                    shader_hash,
                )
        if not self.capabilities.glslang_validator:
            message = "缺少 glslangValidator，无法编译 GLSL 为 SPIR-V"
            self._log(publish, "WARN", label, message)
            return self._certificate(
                target,
                "BLOCKED",
                reference_stage,
                message,
                plan.kernel_id,
                shader_hash,
                reference_hash,
                len(references),
            )

        spirv, last_stage, message = self._compile_shader(shader.source)
        compiled = spirv is not None
        self._log(publish, "INFO" if compiled else "ERROR", label, message)
        if spirv is None:
            if last_stage == "GENERATED":
                last_stage = reference_stage
            return self._certificate(
                target,
                "FAILED",
                last_stage,
                message,
                plan.kernel_id,
                shader_hash,
                reference_hash,
                len(references),
            )

        if not references:
            message = "缺少 onnxruntime，无法生成 ONNX CPU 参考输出"
            self._log(publish, "WARN", label, message)
            return self._certificate(
                target,
                "BLOCKED",
                last_stage,
                message,
                plan.kernel_id,
                shader_hash,
                reference_hash,
            )
        elif not self.capabilities.vulkan_binding:
            message = "缺少 Vulkan Python 绑定，无法提交计算任务"
        elif executor is None:
            message = "Vulkan 执行器初始化失败或不可用"
        if executor is None:
            self._log(publish, "WARN", label, message)
            return self._certificate(
                target,
                "BLOCKED",
                last_stage,
                message,
                plan.kernel_id,
                shader_hash,
                reference_hash,
                len(references),
            )

        passed = 0
        metrics: list[dict[str, object]] = []
        for case_index, reference in enumerate(references, start=1):
            try:
                execution = executor.execute(
                    plan.steps[0],
                    spirv,
                    tuple(reference.feeds.values()),
                    reference.outputs,
                )
                case_passed, case_metrics = _compare_outputs(
                    reference.outputs,
                    execution.outputs,
                )
                metrics.append({"case": case_index, **case_metrics})
                if case_passed:
                    passed += 1
                self._log(
                    publish,
                    "INFO" if case_passed else "ERROR",
                    label,
                    f"GPU 对比 case {case_index}/{len(references)}: "
                    f"{'通过' if case_passed else '失败'} / {case_metrics}",
                )
            except VulkanExecutionError as error:
                message = f"Vulkan 执行失败: {error}"
                self._log(publish, "ERROR", label, message)
                return self._certificate(
                    target,
                    "FAILED",
                    last_stage,
                    message,
                    plan.kernel_id,
                    shader_hash,
                    reference_hash,
                    len(references),
                    passed,
                    metrics,
                )
        confidence = round(passed / len(references), 3) if references else 0.0
        status = "DEVICE_VERIFIED" if passed == len(references) else "FAILED"
        message = f"GPU 数值对比完成: {passed}/{len(references)} cases"
        self._log(publish, "INFO" if status == "DEVICE_VERIFIED" else "ERROR", label, message)
        return self._certificate(
            target,
            status,
            "DEVICE_VERIFIED" if status == "DEVICE_VERIFIED" else last_stage,
            message,
            plan.kernel_id,
            shader_hash,
            reference_hash,
            len(references),
            passed,
            metrics,
            confidence,
        )

    def compile_shader(self, source: str) -> tuple[bytes | None, str, str]:
        """Compile one GLSL compute module with the detected portable toolchain."""
        if not self.capabilities.glslang_validator:
            return None, "GENERATED", "缺少 glslangValidator，无法编译 GLSL 为 SPIR-V"
        with TemporaryDirectory(prefix="ttv-shader-") as directory:
            source_path = Path(directory) / "kernel.comp"
            spirv_path = Path(directory) / "kernel.spv"
            source_path.write_text(source, encoding="utf-8")
            compiler = str(self.capabilities.glslang_validator)
            if compiler.startswith("portable:"):
                command = [
                    compiler.removeprefix("portable:"),
                    str(Path(__file__).parents[4] / "web" / "scripts" / "compile-shader.cjs"),
                    str(source_path),
                    str(spirv_path),
                ]
            else:
                command = [compiler, "-V", "-S", "comp", "-o", str(spirv_path), str(source_path)]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as error:
                return None, "GENERATED", f"GLSL 编译器调用失败: {error}"
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                return None, "GENERATED", f"GLSL 编译失败: {detail}"
            if self.capabilities.spirv_val:
                validated = subprocess.run(
                    [str(self.capabilities.spirv_val), str(spirv_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if validated.returncode != 0:
                    detail = validated.stderr.strip() or validated.stdout.strip()
                    return None, "SPIRV_COMPILED", f"SPIR-V 校验失败: {detail}"
                return spirv_path.read_bytes(), "SPIRV_VALIDATED", "SPIR-V 编译与校验通过"
            return (
                spirv_path.read_bytes(),
                "SPIRV_COMPILED",
                "SPIR-V 编译通过；未找到 spirv-val",
            )

    # Kept for callers written before the compiler entry point became public.
    def _compile_shader(self, source: str) -> tuple[bytes | None, str, str]:
        return self.compile_shader(source)

    @staticmethod
    def _certificate(
        target: VerificationTarget,
        status: str,
        last_completed: str,
        message: str,
        kernel_id: str = "",
        shader_hash: str = "",
        reference_hash: str = "",
        cases_total: int = 0,
        cases_passed: int = 0,
        metrics: list[dict[str, object]] | None = None,
        confidence: float = 0.0,
    ) -> dict[str, object]:
        return {
            "target_id": target.target_id,
            "semantic_key": target.semantic_key,
            "operator": target.op_type,
            "kernel_id": kernel_id,
            "shader_hash": shader_hash,
            "reference_hash": reference_hash,
            "status": status,
            "last_completed": last_completed,
            "confidence": confidence,
            "cases_passed": cases_passed,
            "cases_total": cases_total,
            "metrics": metrics or [],
            "message": message,
        }

    @staticmethod
    def _log(
        publish: Publish,
        level: str,
        operator: str,
        message: str,
    ) -> None:
        publish(
            {
                "type": "log",
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "level": level,
                "operator": operator,
                "message": message,
            }
        )


def _canonical_verification_target(
    target: VerificationTarget,
) -> tuple[VerificationTarget, str | None]:
    """Build a small static sample without changing the verified rank/broadcast class."""
    normalized: VerificationTarget | None = None
    if target.op_type in {"Add", "Mul", "Sub", "Div"}:
        normalized = _canonical_binary_target(target)
    elif target.op_type in {*_FP32_UNARY_OPERATORS, "Cast"}:
        normalized = _canonical_unary_target(target)
    elif target.op_type == "Transpose":
        normalized = _canonical_transpose_target(target)
    elif target.op_type == "Concat":
        normalized = _canonical_concat_target(target)

    if normalized is None or normalized == target:
        return target, None
    return normalized, (
        "规范化验证规格（仅用于 GPU/ORT 小型样本）: "
        f"{_tensor_shapes(target)} -> {_tensor_shapes(normalized)}"
    )


def _canonical_binary_target(target: VerificationTarget) -> VerificationTarget | None:
    if len(target.inputs) != 2 or len(target.outputs) != 1:
        return None
    if any(
        tensor.data_type != "FLOAT"
        for tensor in (*target.inputs, *target.outputs)
    ):
        return None
    if any(not tensor.shape_known for tensor in (*target.inputs, *target.outputs)):
        return None

    output_shape = target.outputs[0].shape
    output_rank = len(output_shape)
    if any(len(tensor.shape) > output_rank for tensor in target.inputs):
        return None

    canonical_output: list[str] = []
    for output_axis, output_dimension in enumerate(output_shape):
        aligned_dimensions = [
            tensor.shape[output_axis - (output_rank - len(tensor.shape))]
            for tensor in target.inputs
            if output_axis >= output_rank - len(tensor.shape)
        ]
        if not _known_broadcast_axis_is_valid(output_dimension, aligned_dimensions):
            return None
        canonical_output.append(
            "1" if output_dimension == "1" else _sample_dimension(output_axis)
        )

    normalized_inputs: list[VerificationTensor] = []
    for tensor in target.inputs:
        rank_offset = output_rank - len(tensor.shape)
        canonical_shape = tuple(
            "1" if dimension == "1" else canonical_output[rank_offset + axis]
            for axis, dimension in enumerate(tensor.shape)
        )
        normalized_inputs.append(replace(tensor, shape=canonical_shape))
    normalized_outputs = (
        replace(target.outputs[0], shape=tuple(canonical_output)),
    )
    return replace(
        target,
        inputs=tuple(normalized_inputs),
        outputs=normalized_outputs,
    )


def _canonical_unary_target(target: VerificationTarget) -> VerificationTarget | None:
    if len(target.inputs) != 1 or len(target.outputs) != 1:
        return None
    source = target.inputs[0]
    output = target.outputs[0]
    if not source.shape_known or not output.shape_known:
        return None
    if target.op_type in _FP32_UNARY_OPERATORS and (
        source.data_type != "FLOAT" or output.data_type != "FLOAT"
    ):
        return None
    if target.op_type == "Cast" and (
        source.data_type,
        output.data_type,
    ) not in {("FLOAT", "INT32"), ("INT32", "FLOAT")}:
        return None
    if len(source.shape) != len(output.shape):
        return None

    canonical_shape: list[str] = []
    for axis, (source_dimension, output_dimension) in enumerate(
        zip(source.shape, output.shape, strict=True)
    ):
        source_value = _positive_literal(source_dimension)
        output_value = _positive_literal(output_dimension)
        if source_dimension.isdigit() and source_value is None:
            return None
        if output_dimension.isdigit() and output_value is None:
            return None
        if (
            source_value is not None
            and output_value is not None
            and source_value != output_value
        ):
            return None
        canonical_shape.append(
            "1"
            if source_dimension == "1" or output_dimension == "1"
            else _sample_dimension(axis)
        )
    shape = tuple(canonical_shape)
    return replace(
        target,
        inputs=(replace(source, shape=shape),),
        outputs=(replace(output, shape=shape),),
    )


def _canonical_transpose_target(
    target: VerificationTarget,
) -> VerificationTarget | None:
    if len(target.inputs) != 1 or len(target.outputs) != 1:
        return None
    source = target.inputs[0]
    output = target.outputs[0]
    if (
        source.data_type != "FLOAT"
        or output.data_type != "FLOAT"
        or not source.shape_known
        or not output.shape_known
        or len(source.shape) != len(output.shape)
    ):
        return None
    rank = len(source.shape)
    raw_permutation = target.attributes.get("perm")
    if raw_permutation is None:
        permutation = tuple(reversed(range(rank)))
    elif (
        isinstance(raw_permutation, list)
        and all(
            isinstance(axis, int) and not isinstance(axis, bool)
            for axis in raw_permutation
        )
    ):
        permutation = tuple(raw_permutation)
    else:
        return None
    if len(permutation) != rank or sorted(permutation) != list(range(rank)):
        return None

    canonical_input = [""] * rank
    for output_axis, input_axis in enumerate(permutation):
        source_dimension = source.shape[input_axis]
        output_dimension = output.shape[output_axis]
        source_value = _positive_literal(source_dimension)
        output_value = _positive_literal(output_dimension)
        if source_dimension.isdigit() and source_value is None:
            return None
        if output_dimension.isdigit() and output_value is None:
            return None
        if (
            source_value is not None
            and output_value is not None
            and source_value != output_value
        ):
            return None
        canonical_input[input_axis] = (
            "1"
            if source_dimension == "1" or output_dimension == "1"
            else _sample_dimension(input_axis)
        )
    canonical_output = tuple(canonical_input[axis] for axis in permutation)
    return replace(
        target,
        inputs=(replace(source, shape=tuple(canonical_input)),),
        outputs=(replace(output, shape=canonical_output),),
    )


def _canonical_concat_target(
    target: VerificationTarget,
) -> VerificationTarget | None:
    if not target.inputs or len(target.outputs) != 1:
        return None
    tensors = (*target.inputs, *target.outputs)
    if any(tensor.data_type != "FLOAT" or not tensor.shape_known for tensor in tensors):
        return None
    output = target.outputs[0]
    rank = len(output.shape)
    if rank == 0 or any(len(tensor.shape) != rank for tensor in target.inputs):
        return None
    raw_axis = target.attributes.get("axis")
    if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
        return None
    axis = raw_axis + rank if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= rank:
        return None

    canonical_non_axis: list[str] = []
    for current_axis, output_dimension in enumerate(output.shape):
        if current_axis == axis:
            canonical_non_axis.append("")
            continue
        input_dimensions = [tensor.shape[current_axis] for tensor in target.inputs]
        known = {
            value
            for dimension in input_dimensions
            if (value := _positive_literal(dimension)) is not None
        }
        output_value = _positive_literal(output_dimension)
        if any(dimension.isdigit() and _positive_literal(dimension) is None for dimension in input_dimensions):
            return None
        if output_dimension.isdigit() and output_value is None:
            return None
        if len(known) > 1 or (known and output_value is not None and output_value not in known):
            return None
        canonical_non_axis.append(
            "1"
            if output_dimension == "1" or any(dimension == "1" for dimension in input_dimensions)
            else _sample_dimension(current_axis)
        )

    canonical_axis_lengths: list[str] = []
    known_axis_lengths: list[int] = []
    all_axis_lengths_known = True
    for index, tensor in enumerate(target.inputs):
        dimension = tensor.shape[axis]
        value = _positive_literal(dimension)
        if dimension.isdigit() and value is None:
            return None
        if value is None:
            all_axis_lengths_known = False
        else:
            known_axis_lengths.append(value)
        canonical_axis_lengths.append("1" if dimension == "1" else str(2 + index % 2))
    output_axis_value = _positive_literal(output.shape[axis])
    if output.shape[axis].isdigit() and output_axis_value is None:
        return None
    if (
        all_axis_lengths_known
        and output_axis_value is not None
        and sum(known_axis_lengths) != output_axis_value
    ):
        return None

    normalized_inputs: list[VerificationTensor] = []
    for tensor, axis_length in zip(target.inputs, canonical_axis_lengths, strict=True):
        shape = list(canonical_non_axis)
        shape[axis] = axis_length
        normalized_inputs.append(replace(tensor, shape=tuple(shape)))
    output_shape = list(canonical_non_axis)
    output_shape[axis] = str(sum(int(value) for value in canonical_axis_lengths))
    return replace(
        target,
        inputs=tuple(normalized_inputs),
        outputs=(replace(output, shape=tuple(output_shape)),),
    )


def _known_broadcast_axis_is_valid(
    output_dimension: str,
    input_dimensions: Sequence[str],
) -> bool:
    output_value = _positive_literal(output_dimension)
    if output_dimension.isdigit() and output_value is None:
        return False
    known_non_unit: set[int] = set()
    for dimension in input_dimensions:
        value = _positive_literal(dimension)
        if dimension.isdigit() and value is None:
            return False
        if value is not None and value != 1:
            known_non_unit.add(value)
    if len(known_non_unit) > 1:
        return False
    return not (
        output_value is not None
        and known_non_unit
        and output_value != next(iter(known_non_unit))
    )


def _positive_literal(dimension: str) -> int | None:
    if not dimension.isdigit():
        return None
    value = int(dimension)
    return value if value > 0 else None


def _sample_dimension(axis: int) -> str:
    return str((2, 3, 4, 5)[axis % 4])


def _tensor_shapes(target: VerificationTarget) -> str:
    def group(tensors: Sequence[VerificationTensor]) -> str:
        return ", ".join(
            f"{tensor.data_type}{list(tensor.shape)}" for tensor in tensors
        )

    return f"inputs({group(target.inputs)}), outputs({group(target.outputs)})"


def _device_name(vulkaninfo: str | None) -> str:
    if not vulkaninfo:
        return "UNAVAILABLE"
    try:
        result = subprocess.run(
            [vulkaninfo, "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    for line in result.stdout.splitlines():
        if "deviceName" in line and "=" in line:
            return line.split("=", 1)[1].strip()
    return "UNKNOWN"


def _portable_glslang() -> str | None:
    node = shutil.which("node") or shutil.which("node.exe")
    helper = Path(__file__).parents[4] / "web" / "scripts" / "compile-shader.cjs"
    package = Path(__file__).parents[4] / "web" / "node_modules" / "@webgpu" / "glslang"
    if node and helper.is_file() and package.is_dir():
        return f"portable:{node}"
    return None


def _run_onnx_reference(
    target: VerificationTarget,
) -> tuple[tuple[ReferenceCase, ...], str, str]:
    import onnxruntime as ort

    # Real graphs may reuse a value name across ports. Isolated reference graphs
    # need unique definition sites even though Vulkan binds buffers by position.
    input_names = [f"input_{index}" for index in range(len(target.inputs))]
    output_names = [f"output_{index}" for index in range(len(target.outputs))]
    input_infos = [
        helper.make_tensor_value_info(name, _onnx_data_type(item.data_type), _shape(item.shape))
        for name, item in zip(input_names, target.inputs, strict=True)
    ]
    output_infos = [
        helper.make_tensor_value_info(name, _onnx_data_type(item.data_type), _shape(item.shape))
        for name, item in zip(output_names, target.outputs, strict=True)
    ]
    node = helper.make_node(
        target.op_type,
        input_names,
        output_names,
        domain=target.domain,
        **target.attributes,
    )
    graph = helper.make_graph([node], f"verify_{target.op_type}", input_infos, output_infos)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(target.domain, target.opset_version)],
    )
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    digest = hashlib.sha256()
    references: list[ReferenceCase] = []
    for case_index in range(3):
        rng = np.random.default_rng(20260801 + case_index)
        feeds: dict[str, np.ndarray[Any, Any]] = {}
        for index, (name, tensor) in enumerate(zip(input_names, target.inputs, strict=True)):
            shape = tuple(_shape(tensor.shape))
            if tensor.data_type == "FLOAT":
                if case_index == 1:
                    value = np.zeros(shape, dtype=np.float32)
                else:
                    value = rng.uniform(-3.0, 3.0, size=shape).astype(np.float32)
                if target.op_type == "Sqrt":
                    value = np.abs(value)
                if target.op_type == "Div" and index == 1:
                    value = np.where(np.abs(value) < 0.25, np.float32(0.25), value)
                if target.op_type == "Div" and case_index == 1 and index == 1:
                    value = np.ones(shape, dtype=np.float32)
            elif tensor.data_type == "INT32":
                value = rng.integers(-32, 33, size=shape, dtype=np.int32)
            else:
                raise ValueError(f"参考执行暂不支持输入类型 {tensor.data_type}")
            feeds[name] = value
        outputs = tuple(np.asarray(output) for output in session.run(output_names, feeds))
        case_digest = hashlib.sha256()
        summaries: list[str] = []
        for output in outputs:
            case_digest.update(output.dtype.str.encode())
            case_digest.update(str(output.shape).encode())
            case_digest.update(output.tobytes())
            summaries.append(f"{output.dtype}{list(output.shape)}")
        digest.update(case_digest.digest())
        references.append(
            ReferenceCase(feeds, outputs, case_digest.hexdigest(), ", ".join(summaries))
        )
    return tuple(references), digest.hexdigest(), references[0].summary


def _compare_outputs(
    references: Sequence[np.ndarray[Any, Any]],
    actual: Sequence[np.ndarray[Any, Any]],
) -> tuple[bool, dict[str, object]]:
    if len(references) != len(actual):
        return False, {"reason": f"输出数量 {len(actual)} != {len(references)}"}
    max_abs = 0.0
    max_rel = 0.0
    for expected, observed in zip(references, actual, strict=True):
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            return False, {
                "reason": (
                    f"输出规格不匹配: expected {expected.dtype}{expected.shape}, "
                    f"actual {observed.dtype}{observed.shape}"
                )
            }
        if np.issubdtype(expected.dtype, np.floating):
            difference = np.abs(observed - expected)
            denominator = np.maximum(np.abs(expected), np.float32(1e-12))
            max_abs = max(max_abs, float(np.max(difference, initial=0.0)))
            max_rel = max(max_rel, float(np.max(difference / denominator, initial=0.0)))
            if not np.allclose(observed, expected, rtol=1e-5, atol=1e-6, equal_nan=True):
                return False, {
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "rtol": 1e-5,
                    "atol": 1e-6,
                }
        elif not np.array_equal(observed, expected):
            return False, {"reason": "整数输出存在不一致"}
    return True, {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "rtol": 1e-5,
        "atol": 1e-6,
    }


def _onnx_data_type(data_type: str) -> int:
    values = {
        "FLOAT": TensorProto.FLOAT,
        "INT32": TensorProto.INT32,
    }
    if data_type not in values:
        raise ValueError(f"参考执行暂不支持张量类型 {data_type}")
    return values[data_type]


def _shape(shape: tuple[str, ...]) -> list[int]:
    if any(not dimension.isdigit() for dimension in shape):
        raise ValueError(f"参考执行要求静态形状: {shape}")
    return [int(dimension) for dimension in shape]
