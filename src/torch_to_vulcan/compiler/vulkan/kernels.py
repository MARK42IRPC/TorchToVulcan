"""Conservative first-pass Vulkan kernel candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ..contracts import OperatorCapability
from .ir import BufferBinding, DispatchPlan, DispatchStep, ShaderModule

MAX_DISPATCH_GROUP_COUNT = 65_535
MAX_SHADER_ELEMENTS = (1 << 32) - 1


class UnsupportedKernel(ValueError):
    """The operator is known, but this candidate does not cover its context."""


@dataclass(frozen=True, slots=True)
class KernelTensor:
    name: str
    data_type: str
    shape: tuple[str, ...]
    shape_known: bool = True
    layout: str = "contiguous"
    strides: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class KernelContext:
    domain: str
    op_type: str
    opset_version: int
    attributes: dict[str, object]
    inputs: tuple[KernelTensor, ...]
    outputs: tuple[KernelTensor, ...]
    layout: str = "contiguous"
    constant_inputs: Mapping[str, object] | None = None
    runtime_inputs: tuple[KernelTensor, ...] | None = None


Builder = Callable[[KernelContext], DispatchPlan]


@dataclass(frozen=True, slots=True)
class KernelCandidate:
    kernel_id: str
    domain: str
    op_type: str
    builder: Builder
    capability: OperatorCapability | None = None


class KernelRegistry:
    def __init__(self) -> None:
        self._candidates: list[KernelCandidate] = []

    def register(self, candidate: KernelCandidate) -> None:
        self._candidates.append(candidate)

    def capabilities(self) -> tuple[OperatorCapability, ...]:
        """Return the declared support matrix of this backend registry."""
        return tuple(self._candidate_capability(candidate) for candidate in self._candidates)

    def select(self, context: KernelContext) -> DispatchPlan:
        candidates = [
            candidate
            for candidate in self._candidates
            if candidate.domain == context.domain and candidate.op_type == context.op_type
        ]
        if not candidates:
            raise UnsupportedKernel("尚未注册 Vulkan Kernel Candidate")
        reasons: list[str] = []
        for candidate in candidates:
            capability = self._candidate_capability(candidate)
            if not capability.matches(
                domain=context.domain,
                op_type=context.op_type,
                opset_version=context.opset_version,
                data_types=tuple(
                    tensor.data_type for tensor in (*context.inputs, *context.outputs)
                ),
                layout=context.layout,
            ):
                reasons.append(f"{candidate.kernel_id}: capability contract mismatch")
                continue
            try:
                return candidate.builder(context)
            except UnsupportedKernel as error:
                reasons.append(f"{candidate.kernel_id}: {error}")
        raise UnsupportedKernel("；".join(reasons))

    @staticmethod
    def _candidate_capability(candidate: KernelCandidate) -> OperatorCapability:
        if candidate.capability is not None:
            return candidate.capability
        data_types = (
            frozenset({"*"})
            if candidate.op_type in {"Identity", "Reshape"}
            else frozenset({"FLOAT", "INT32"})
            if candidate.op_type == "Cast"
            else frozenset({"FLOAT"})
        )
        return OperatorCapability(
            domain=candidate.domain,
            op_type=candidate.op_type,
            data_types=data_types,
            notes=(f"kernel_id={candidate.kernel_id}",),
        )


def default_kernel_registry() -> KernelRegistry:
    registry = KernelRegistry()
    for op_type, symbol in (("Add", "+"), ("Mul", "*"), ("Sub", "-"), ("Div", "/")):
        registry.register(
            KernelCandidate(
                kernel_id=f"elementwise.{op_type.lower()}.fp32",
                domain="",
                op_type=op_type,
                builder=lambda context, symbol=symbol: _binary_fp32(context, symbol),
            )
        )
    registry.register(
        KernelCandidate("elementwise.relu.fp32", "", "Relu", _relu_fp32)
    )
    for op_type, expression in (
        ("Neg", "-value"),
        ("Exp", "exp(value)"),
        ("Floor", "floor(value)"),
        ("Sin", "sin(value)"),
        ("Cos", "cos(value)"),
        ("Sqrt", "sqrt(value)"),
        ("Tanh", "tanh(value)"),
        ("Sigmoid", "1.0 / (1.0 + exp(-value))"),
    ):
        registry.register(
            KernelCandidate(
                f"elementwise.{op_type.lower()}.fp32",
                "",
                op_type,
                builder=lambda context, expression=expression: _unary_fp32(
                    context, expression
                ),
            )
        )
    registry.register(
        KernelCandidate(
            "elementwise.leaky_relu.fp32", "", "LeakyRelu", _leaky_relu_fp32
        )
    )
    registry.register(KernelCandidate("cast.fp32-int32", "", "Cast", _cast))
    registry.register(
        KernelCandidate("layout.transpose.fp32", "", "Transpose", _transpose_fp32)
    )
    registry.register(KernelCandidate("layout.concat.fp32", "", "Concat", _concat_fp32))
    registry.register(
        KernelCandidate(
            "linear.matmul.fp32",
            "",
            "MatMul",
            _matmul_fp32,
            capability=OperatorCapability(
                domain="",
                op_type="MatMul",
                data_types=frozenset({"FLOAT"}),
                layouts=frozenset({"contiguous"}),
                notes=("静态 FP32；支持 rank>=2 和 trailing batch broadcast",),
            ),
        )
    )
    registry.register(
        KernelCandidate(
            "linear.gemm.fp32",
            "",
            "Gemm",
            _gemm_fp32,
            capability=OperatorCapability(
                domain="",
                op_type="Gemm",
                data_types=frozenset({"FLOAT"}),
                layouts=frozenset({"contiguous"}),
                notes=("二维 FP32；常见 C 广播形式",),
            ),
        )
    )
    registry.register(
        KernelCandidate(
            "reduction.reduce_mean.fp32",
            "",
            "ReduceMean",
            _reduce_mean_fp32,
            capability=OperatorCapability(
                domain="",
                op_type="ReduceMean",
                data_types=frozenset({"FLOAT", "INT64"}),
                layouts=frozenset({"contiguous"}),
                notes=("静态 FP32；常量 axes；连续 reduction axes",),
            ),
        )
    )
    registry.register(
        KernelCandidate(
            "normalization.softmax.fp32",
            "",
            "Softmax",
            _softmax_fp32,
            capability=OperatorCapability(
                domain="",
                op_type="Softmax",
                data_types=frozenset({"FLOAT"}),
                layouts=frozenset({"contiguous"}),
                notes=("静态 FP32；任意合法 axis",),
            ),
        )
    )
    registry.register(
        KernelCandidate(
            "normalization.layer_norm.fp32",
            "",
            "LayerNormalization",
            _layer_normalization_fp32,
            capability=OperatorCapability(
                domain="",
                op_type="LayerNormalization",
                data_types=frozenset({"FLOAT"}),
                layouts=frozenset({"contiguous"}),
                notes=("静态 FP32；trailing normalized dimensions；仅输出 Y",),
            ),
        )
    )
    registry.register(KernelCandidate("view.reshape", "", "Reshape", _metadata_view))
    registry.register(KernelCandidate("view.identity", "", "Identity", _metadata_view))
    return registry


def _binary_fp32(context: KernelContext, symbol: str) -> DispatchPlan:
    _require_arity(context, 2, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    element_count, input_indices = _require_broadcast_static_fp32_shape(context)
    operation = (
        f"input_a[{input_indices[0]}] {symbol} input_b[{input_indices[1]}]"
    )
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputA { float input_a[]; };",
            "layout(set = 0, binding = 1, std430) readonly buffer InputB { float input_b[]; };",
            "layout(set = 0, binding = 2, std430) writeonly buffer Output { float output_y[]; };",
        ),
        f"output_y[index] = {operation};",
    )
    return _single_dispatch_plan(context, source, element_count, input_count=2)


def _relu_fp32(context: KernelContext) -> DispatchPlan:
    _require_arity(context, 1, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    element_count = _require_same_static_fp32_shape(context)
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputX { float input_x[]; };",
            "layout(set = 0, binding = 1, std430) writeonly buffer Output { float output_y[]; };",
        ),
        "output_y[index] = max(input_x[index], 0.0);",
    )
    return _single_dispatch_plan(context, source, element_count, input_count=1)


def _unary_fp32(context: KernelContext, expression: str) -> DispatchPlan:
    _require_arity(context, 1, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    element_count = _require_same_static_fp32_shape(context)
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputX { float input_x[]; };",
            "layout(set = 0, binding = 1, std430) writeonly buffer Output { float output_y[]; };",
        ),
        f"float value = input_x[index];\n    output_y[index] = {expression};",
    )
    return _single_dispatch_plan(context, source, element_count, input_count=1)


def _leaky_relu_fp32(context: KernelContext) -> DispatchPlan:
    alpha_value = context.attributes.get("alpha", 0.01)
    if isinstance(alpha_value, bool) or not isinstance(alpha_value, (int, float)):
        raise UnsupportedKernel("LeakyRelu alpha 必须是有限浮点数")
    alpha = float(alpha_value)
    if not math.isfinite(alpha):
        raise UnsupportedKernel("LeakyRelu alpha 必须是有限浮点数")
    return _unary_fp32(
        context,
        f"value >= 0.0 ? value : value * {_glsl_float(alpha)}",
    )


def _cast(context: KernelContext) -> DispatchPlan:
    _require_arity(context, 1, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    input_tensor = context.inputs[0]
    output_tensor = context.outputs[0]
    if input_tensor.shape != output_tensor.shape:
        raise UnsupportedKernel("Cast 输入输出形状必须一致")
    element_count = _static_element_count(output_tensor.shape)
    conversions = {
        ("FLOAT", "INT32"): ("float", "int", "int(input_x[index])"),
        ("INT32", "FLOAT"): ("int", "float", "float(input_x[index])"),
    }
    conversion = conversions.get((input_tensor.data_type, output_tensor.data_type))
    if conversion is None:
        raise UnsupportedKernel(
            f"首版仅支持 FLOAT↔INT32，当前为 {input_tensor.data_type}→{output_tensor.data_type}"
        )
    input_glsl, output_glsl, expression = conversion
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputX "
            f"{{ {input_glsl} input_x[]; }};",
            "layout(set = 0, binding = 1, std430) writeonly buffer Output "
            f"{{ {output_glsl} output_y[]; }};",
        ),
        f"output_y[index] = {expression};",
    )
    return _single_dispatch_plan(context, source, element_count, input_count=1)


def _transpose_fp32(context: KernelContext) -> DispatchPlan:
    _require_arity(context, 1, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    source_tensor = context.inputs[0]
    output_tensor = context.outputs[0]
    if source_tensor.data_type != "FLOAT" or output_tensor.data_type != "FLOAT":
        raise UnsupportedKernel(
            "首版 Transpose 仅支持 FP32，当前为 "
            f"{source_tensor.data_type}→{output_tensor.data_type}"
        )
    if len(source_tensor.shape) != len(output_tensor.shape):
        raise UnsupportedKernel("Transpose 输入输出秩必须一致")
    rank = len(source_tensor.shape)
    permutation = _transpose_permutation(context.attributes.get("perm"), rank)
    input_shape = _static_shape(source_tensor.shape)
    output_shape = _static_shape(output_tensor.shape)
    inferred_shape = tuple(input_shape[axis] for axis in permutation)
    if inferred_shape != output_shape:
        raise UnsupportedKernel(
            f"Transpose 输出形状不匹配: 推导为 {inferred_shape}，实际为 {output_shape}"
        )
    element_count = _static_element_count(output_tensor.shape)
    input_index = _transpose_input_index(input_shape, output_shape, permutation)
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputX { float input_x[]; };",
            "layout(set = 0, binding = 1, std430) writeonly buffer Output { float output_y[]; };",
        ),
        f"output_y[index] = input_x[{input_index}];",
    )
    return _single_dispatch_plan(
        context,
        source,
        element_count,
        input_count=1,
        kernel_id="layout.transpose.fp32",
    )


def _concat_fp32(context: KernelContext) -> DispatchPlan:
    if not context.inputs or len(context.outputs) != 1:
        raise UnsupportedKernel(
            f"Concat 期望至少 1 个输入/1 个输出，实际为 {len(context.inputs)}/{len(context.outputs)}"
        )
    _require_known_shapes((*context.inputs, *context.outputs))
    if any(
        tensor.data_type != "FLOAT"
        for tensor in (*context.inputs, *context.outputs)
    ):
        types = ", ".join(
            tensor.data_type for tensor in (*context.inputs, *context.outputs)
        )
        raise UnsupportedKernel(f"首版 Concat 仅支持 FP32，当前类型为 {types}")

    output_shape = _static_shape(context.outputs[0].shape)
    rank = len(output_shape)
    axis = _normalize_axis(context.attributes.get("axis"), rank, "Concat")
    input_shapes = tuple(_static_shape(tensor.shape) for tensor in context.inputs)
    if any(len(shape) != rank for shape in input_shapes):
        raise UnsupportedKernel("Concat 所有输入输出秩必须一致")
    for shape in input_shapes:
        if any(
            dimension != output_shape[index]
            for index, dimension in enumerate(shape)
            if index != axis
        ):
            raise UnsupportedKernel("Concat 非连接轴尺寸必须一致")
    axis_lengths = tuple(shape[axis] for shape in input_shapes)
    if sum(axis_lengths) != output_shape[axis]:
        raise UnsupportedKernel(
            f"Concat 连接轴长度之和 {sum(axis_lengths)} != 输出 {output_shape[axis]}"
        )

    inner = math.prod(output_shape[axis + 1 :])
    output_axis = output_shape[axis]
    prefix = 0
    branches: list[str] = []
    for index, axis_length in enumerate(axis_lengths):
        condition = (
            f"if (axis_index < {prefix + axis_length}u)"
            if index == 0
            else f"else if (axis_index < {prefix + axis_length}u)"
        )
        local_axis = (
            "axis_index" if prefix == 0 else f"(axis_index - {prefix}u)"
        )
        input_axis_block = axis_length * inner
        branches.append(
            f"{condition} {{\n"
            f"        uint source_index = outer_index * {input_axis_block}u "
            f"+ {local_axis} * {inner}u + inner_index;\n"
            f"        output_y[index] = input_{index}[source_index];\n"
            "    }"
        )
        prefix += axis_length
    operation = (
        f"uint inner_index = index % {inner}u;\n"
        f"    uint axis_index = (index / {inner}u) % {output_axis}u;\n"
        f"    uint outer_index = index / {output_axis * inner}u;\n"
        f"    {' '.join(branches)}"
    )
    declarations = tuple(
        "layout(set = 0, binding = "
        f"{index}, std430) readonly buffer Input{index} {{ float input_{index}[]; }};"
        for index in range(len(context.inputs))
    ) + (
        "layout(set = 0, binding = "
        f"{len(context.inputs)}, std430) writeonly buffer Output {{ float output_y[]; }};",
    )
    source = _shader_source(declarations, operation)
    return _single_dispatch_plan(
        context,
        source,
        _static_element_count(context.outputs[0].shape),
        input_count=len(context.inputs),
        kernel_id="layout.concat.fp32",
    )


def _matmul_fp32(context: KernelContext) -> DispatchPlan:
    _require_arity(context, 2, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    a, b = context.inputs
    output = context.outputs[0]
    _require_fp32(context)
    a_shape = _static_shape(a.shape)
    b_shape = _static_shape(b.shape)
    output_shape = _static_shape(output.shape)
    if len(a_shape) < 2 or len(b_shape) < 2 or len(output_shape) < 2:
        raise UnsupportedKernel("MatMul 首版要求两个输入和输出的秩至少为 2")
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    output_batch = output_shape[:-2]
    inferred_batch = _broadcast_shape((a_batch, b_batch))
    m, k = a_shape[-2:]
    k_b, n = b_shape[-2:]
    if k != k_b or output_shape != (*inferred_batch, m, n):
        raise UnsupportedKernel(
            f"MatMul 形状不匹配: {a_shape} @ {b_shape} -> {output_shape}; "
            f"期望 batch={inferred_batch}, matrix=({m}, {n})"
        )
    output_element_count = _static_element_count(output.shape)
    matrix_element_count = m * n
    a_batch_offset = _matmul_batch_offset(a_batch, output_batch, m * k)
    b_batch_offset = _matmul_batch_offset(b_batch, output_batch, k * n)
    a_element_index = (
        f"row * {k}u + inner"
        if not a_batch
        else f"a_batch_offset + row * {k}u + inner"
    )
    b_element_index = (
        f"inner * {n}u + column"
        if not b_batch
        else f"b_batch_offset + inner * {n}u + column"
    )
    operation = (
        f"float sum = 0.0;\n"
        f"    for (uint inner = 0u; inner < {k}u; ++inner) {{\n"
        f"        sum += input_a[{a_element_index}] * input_b[{b_element_index}];\n"
        f"    }}\n"
        "    output_y[index] = sum;"
    )
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputA { float input_a[]; };",
            "layout(set = 0, binding = 1, std430) readonly buffer InputB { float input_b[]; };",
            "layout(set = 0, binding = 2, std430) writeonly buffer Output { float output_y[]; };",
        ),
        "uint batch_index = index / " + str(matrix_element_count) + "u;\n"
        "    uint row = (index / " + str(n) + "u) % " + str(m) + "u;\n"
        "    uint column = index % " + str(n) + "u;\n"
        + f"    uint a_batch_offset = {a_batch_offset};\n"
        + f"    uint b_batch_offset = {b_batch_offset};\n"
        + operation,
    )
    return _single_dispatch_plan(
        context,
        source,
        output_element_count,
        input_count=2,
        kernel_id="linear.matmul.fp32",
    )


def _matmul_batch_offset(
    input_batch: tuple[int, ...],
    output_batch: tuple[int, ...],
    matrix_element_count: int,
) -> str:
    """Map one flattened output batch index to an input element offset."""
    if not input_batch:
        return "0u"
    rank_offset = len(output_batch) - len(input_batch)
    input_strides = _row_major_strides(input_batch)
    output_strides = _row_major_strides(output_batch)
    terms: list[str] = []
    for input_axis, (dimension, input_stride) in enumerate(
        zip(input_batch, input_strides, strict=True)
    ):
        if dimension == 1:
            continue
        output_axis = rank_offset + input_axis
        output_dimension = output_batch[output_axis]
        output_stride = output_strides[output_axis]
        if output_stride == 1:
            coordinate = f"(batch_index % {output_dimension}u)"
        else:
            coordinate = (
                f"((batch_index / {output_stride}u) % {output_dimension}u)"
            )
        if input_stride != 1:
            coordinate += f" * {input_stride}u"
        terms.append(coordinate)
    if not terms:
        return "0u"
    return f"({ ' + '.join(terms) }) * {matrix_element_count}u"


def _reduce_mean_fp32(context: KernelContext) -> DispatchPlan:
    if len(context.inputs) not in {1, 2} or len(context.outputs) != 1:
        raise UnsupportedKernel(
            "ReduceMean 期望 1 个 X、可选 axes 和 1 个输出，实际为 "
            f"{len(context.inputs)} 个输入/{len(context.outputs)} 个输出"
        )
    _require_known_shapes((*context.inputs, *context.outputs))
    _require_fp32(
        KernelContext(
            context.domain,
            context.op_type,
            context.opset_version,
            context.attributes,
            (context.inputs[0],),
            context.outputs,
            context.layout,
            context.constant_inputs,
            (context.inputs[0],),
        )
    )
    input_shape = _static_shape(context.inputs[0].shape)
    output_shape = _static_shape(context.outputs[0].shape)
    axes = _reduce_axes(context, len(input_shape))
    keepdims = _attribute_flag(context.attributes.get("keepdims", 1), "ReduceMean keepdims")
    if context.attributes.get("noop_with_empty_axes", 0) not in {0, False}:
        raise UnsupportedKernel("ReduceMean noop_with_empty_axes=1 暂未支持")
    expected_shape = _reduction_output_shape(input_shape, axes, keepdims)
    if output_shape != expected_shape:
        raise UnsupportedKernel(
            f"ReduceMean 输出形状不匹配: axes={axes}, keepdims={int(keepdims)}, "
            f"期望 {expected_shape}，实际 {output_shape}"
        )
    reduction_count = math.prod(input_shape[axis] for axis in axes)
    output_count = _static_element_count(context.outputs[0].shape)
    input_strides = _row_major_strides(input_shape)
    output_strides = _row_major_strides(output_shape)
    output_axis_for_input = {
        input_axis: (
            input_axis
            if keepdims
            else sum(1 for axis in range(input_axis) if axis not in axes)
        )
        for input_axis in range(len(input_shape))
        if input_axis not in axes
    }
    fixed_terms: list[str] = []
    for input_axis, input_stride in enumerate(input_strides):
        if input_axis in axes:
            continue
        output_axis = output_axis_for_input[input_axis]
        output_dimension = output_shape[output_axis]
        output_stride = output_strides[output_axis]
        if output_dimension == 1 or output_stride == 1:
            coordinate = f"(index % {output_dimension}u)" if output_stride == 1 else f"((index / {output_stride}u) % {output_dimension}u)"
        else:
            coordinate = f"((index / {output_stride}u) % {output_dimension}u)"
        if input_stride != 1:
            coordinate += f" * {input_stride}u"
        fixed_terms.append(coordinate)
    fixed_offset = " + ".join(fixed_terms) if fixed_terms else "0u"
    reduced_terms: list[str] = []
    for reduced_index, input_axis in enumerate(axes):
        divisor = math.prod(input_shape[axis] for axis in axes[reduced_index + 1 :])
        coordinate = (
            f"((reduction / {divisor}u) % {input_shape[input_axis]}u)"
        )
        input_stride = input_strides[input_axis]
        if input_stride != 1:
            coordinate += f" * {input_stride}u"
        reduced_terms.append(coordinate)
    reduced_offset = " + ".join(reduced_terms) if reduced_terms else "0u"
    runtime_context = KernelContext(
        context.domain,
        context.op_type,
        context.opset_version,
        context.attributes,
        (context.inputs[0],),
        context.outputs,
        context.layout,
        context.constant_inputs,
        (context.inputs[0],),
    )
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputX { float input_x[]; };",
            "layout(set = 0, binding = 1, std430) writeonly buffer Output { float output_y[]; };",
        ),
        f"uint input_base = {fixed_offset};\n"
        f"    float sum = 0.0;\n"
        f"    for (uint reduction = 0u; reduction < {reduction_count}u; ++reduction) {{\n"
        f"        sum += input_x[input_base + {reduced_offset}];\n"
        "    }\n"
        f"    output_y[index] = sum / {_glsl_float(float(reduction_count))};",
    )
    return _single_dispatch_plan(
        runtime_context,
        source,
        output_count,
        input_count=1,
        kernel_id="reduction.reduce_mean.fp32",
    )


def _softmax_fp32(context: KernelContext) -> DispatchPlan:
    _require_arity(context, 1, 1)
    _require_known_shapes((*context.inputs, *context.outputs))
    _require_fp32(context)
    input_shape = _static_shape(context.inputs[0].shape)
    output_shape = _static_shape(context.outputs[0].shape)
    if input_shape != output_shape:
        raise UnsupportedKernel("Softmax 输入输出形状必须一致")
    axis = _normalize_axis(context.attributes.get("axis", -1), len(input_shape), "Softmax")
    axis_length = input_shape[axis]
    inner = math.prod(input_shape[axis + 1 :])
    slice_length = axis_length * inner
    element_count = _static_element_count(context.outputs[0].shape)
    source = _shader_source(
        (
            "layout(set = 0, binding = 0, std430) readonly buffer InputX { float input_x[]; };",
            "layout(set = 0, binding = 1, std430) writeonly buffer Output { float output_y[]; };",
        ),
        f"uint axis_index = (index / {inner}u) % {axis_length}u;\n"
        f"    uint inner_index = index % {inner}u;\n"
        f"    uint slice_base = (index / {slice_length}u) * {slice_length}u + inner_index;\n"
        "    float maximum = -3.402823466e+38;\n"
        f"    for (uint axis_value = 0u; axis_value < {axis_length}u; ++axis_value) {{\n"
        f"        maximum = max(maximum, input_x[slice_base + axis_value * {inner}u]);\n"
        "    }\n"
        "    float denominator = 0.0;\n"
        f"    for (uint axis_value = 0u; axis_value < {axis_length}u; ++axis_value) {{\n"
        f"        denominator += exp(input_x[slice_base + axis_value * {inner}u] - maximum);\n"
        "    }\n"
        f"    output_y[index] = exp(input_x[slice_base + axis_index * {inner}u] - maximum) / denominator;",
    )
    return _single_dispatch_plan(
        context,
        source,
        element_count,
        input_count=1,
        kernel_id="normalization.softmax.fp32",
    )


def _layer_normalization_fp32(context: KernelContext) -> DispatchPlan:
    if len(context.inputs) not in {2, 3} or len(context.outputs) != 1:
        raise UnsupportedKernel(
            "LayerNormalization 首版要求 X/Scale/[B] 三个以内输入且仅输出 Y"
        )
    _require_known_shapes((*context.inputs, *context.outputs))
    _require_fp32(context)
    x, scale = context.inputs[:2]
    bias = context.inputs[2] if len(context.inputs) == 3 else None
    x_shape = _static_shape(x.shape)
    output_shape = _static_shape(context.outputs[0].shape)
    axis = _normalize_axis(
        context.attributes.get("axis", -1), len(x_shape), "LayerNormalization"
    )
    if output_shape != x_shape:
        raise UnsupportedKernel(
            "LayerNormalization 输出形状必须与 X 一致"
        )
    normalized_shape = x_shape[axis:]
    if _static_shape(scale.shape) != normalized_shape:
        raise UnsupportedKernel(
            f"LayerNormalization Scale 形状 {_static_shape(scale.shape)} != "
            f"normalized shape {normalized_shape}"
        )
    if bias is not None and _static_shape(bias.shape) != normalized_shape:
        raise UnsupportedKernel(
            f"LayerNormalization Bias 形状 {_static_shape(bias.shape)} != "
            f"normalized shape {normalized_shape}"
        )
    stash_type = context.attributes.get("stash_type", 1)
    if isinstance(stash_type, bool) or not isinstance(stash_type, int) or stash_type != 1:
        raise UnsupportedKernel("LayerNormalization 首版仅支持 stash_type=1")
    epsilon = _finite_float(
        context.attributes.get("epsilon", 1e-5), "LayerNormalization epsilon"
    )
    if epsilon < 0.0:
        raise UnsupportedKernel("LayerNormalization epsilon 必须非负")
    normalized_count = math.prod(normalized_shape)
    element_count = _static_element_count(context.outputs[0].shape)
    bias_binding = 2 if bias is not None else None
    output_binding = 3 if bias is not None else 2
    declarations = [
        "layout(set = 0, binding = 0, std430) readonly buffer InputX { float input_x[]; };",
        "layout(set = 0, binding = 1, std430) readonly buffer InputScale { float input_scale[]; };",
    ]
    if bias is not None:
        declarations.append(
            "layout(set = 0, binding = 2, std430) readonly buffer InputBias { float input_bias[]; };"
        )
    declarations.append(
        f"layout(set = 0, binding = {output_binding}, std430) writeonly buffer Output {{ float output_y[]; }};"
    )
    bias_expression = (
        "0.0"
        if bias_binding is None
        else "input_bias[index % " + str(normalized_count) + "u]"
    )
    group_size = normalized_count
    source = _shader_source(
        tuple(declarations),
        f"uint normalized_index = index % {group_size}u;\n"
        f"uint group_base = (index / {group_size}u) * {group_size}u;\n"
        "float mean = 0.0;\n"
        f"for (uint normalized = 0u; normalized < {group_size}u; ++normalized) {{\n"
        "    mean += input_x[group_base + normalized];\n"
        "}\n"
        f"mean /= {_glsl_float(float(group_size))};\n"
        "float variance = 0.0;\n"
        f"for (uint normalized = 0u; normalized < {group_size}u; ++normalized) {{\n"
        "    float centered = input_x[group_base + normalized] - mean;\n"
        "    variance += centered * centered;\n"
        "}\n"
        f"variance /= {_glsl_float(float(group_size))};\n"
        f"float normalized_value = (input_x[index] - mean) * inversesqrt(variance + {_glsl_float(epsilon)});\n"
        f"output_y[index] = normalized_value * input_scale[normalized_index] + {bias_expression};",
    )
    return _single_dispatch_plan(
        context,
        source,
        element_count,
        input_count=len(context.inputs),
        kernel_id="normalization.layer_norm.fp32",
    )


def _reduce_axes(context: KernelContext, rank: int) -> tuple[int, ...]:
    value: object = context.attributes.get("axes")
    if value is None and len(context.inputs) == 2:
        if not context.constant_inputs:
            raise UnsupportedKernel("ReduceMean axes 必须是编译期常量")
        value = context.constant_inputs.get(context.inputs[1].name)
        if value is None:
            raise UnsupportedKernel("ReduceMean axes 必须是编译期常量")
    if value is None and context.constant_inputs:
        value = context.constant_inputs.get("axes")
    if value is None:
        axes = tuple(range(rank))
    else:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, int) and not isinstance(value, bool):
            values = (value,)
        elif isinstance(value, (list, tuple)):
            values = tuple(value)
        else:
            raise UnsupportedKernel("ReduceMean axes 必须是编译期整数数组")
        if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in values):
            raise UnsupportedKernel("ReduceMean axes 必须是编译期整数数组")
        axes = tuple(int(axis) + rank if int(axis) < 0 else int(axis) for axis in values)
    if not axes and rank > 0:
        axes = tuple(range(rank))
    if not axes:
        return ()
    if len(set(axes)) != len(axes) or any(axis < 0 or axis >= rank for axis in axes):
        raise UnsupportedKernel(f"ReduceMean axes 越界或重复: {axes}")
    ordered = tuple(sorted(axes))
    if ordered != tuple(range(ordered[0], ordered[-1] + 1)):
        raise UnsupportedKernel("ReduceMean 首版只支持连续 reduction axes")
    return ordered


def _reduction_output_shape(
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
) -> tuple[int, ...]:
    if keepdims:
        return tuple(1 if axis in axes else dimension for axis, dimension in enumerate(input_shape))
    return tuple(dimension for axis, dimension in enumerate(input_shape) if axis not in axes)


def _gemm_fp32(context: KernelContext) -> DispatchPlan:
    if len(context.inputs) not in {2, 3} or len(context.outputs) != 1:
        raise UnsupportedKernel(
            f"Gemm 期望 2 或 3 个输入/1 个输出，实际为 "
            f"{len(context.inputs)}/{len(context.outputs)}"
        )
    _require_known_shapes((*context.inputs, *context.outputs))
    _require_fp32(context)
    trans_a = _attribute_flag(context.attributes.get("transA", 0), "transA")
    trans_b = _attribute_flag(context.attributes.get("transB", 0), "transB")
    alpha = _finite_float(context.attributes.get("alpha", 1.0), "alpha")
    beta = _finite_float(context.attributes.get("beta", 1.0), "beta")
    a_shape = _static_shape(context.inputs[0].shape)
    b_shape = _static_shape(context.inputs[1].shape)
    output_shape = _static_shape(context.outputs[0].shape)
    if len(a_shape) != 2 or len(b_shape) != 2 or len(output_shape) != 2:
        raise UnsupportedKernel("首版 Gemm 仅支持二维矩阵")
    m = a_shape[1] if trans_a else a_shape[0]
    k = a_shape[0] if trans_a else a_shape[1]
    k_b = b_shape[1] if trans_b else b_shape[0]
    n = b_shape[0] if trans_b else b_shape[1]
    if k != k_b or output_shape != (m, n):
        raise UnsupportedKernel(
            f"Gemm 形状不匹配: A={a_shape}, B={b_shape}, Y={output_shape}, "
            f"transA={int(trans_a)}, transB={int(trans_b)}"
        )
    c_index = "0u"
    declarations = [
        "layout(set = 0, binding = 0, std430) readonly buffer InputA { float input_a[]; };",
        "layout(set = 0, binding = 1, std430) readonly buffer InputB { float input_b[]; };",
    ]
    if len(context.inputs) == 3:
        c_shape = _static_shape(context.inputs[2].shape)
        c_index = _gemm_c_index(c_shape, m, n)
        declarations.append(
            "layout(set = 0, binding = 2, std430) readonly buffer InputC { float input_c[]; };"
        )
    output_binding = len(declarations)
    declarations.append(
        "layout(set = 0, binding = "
        f"{output_binding}, std430) writeonly buffer Output {{ float output_y[]; }};"
    )
    a_index = f"(trans_a ? inner * {m}u + row : row * {k}u + inner)"
    b_index = f"(trans_b ? column * {k}u + inner : inner * {n}u + column)"
    c_term = ""
    if len(context.inputs) == 3:
        c_term = f" + {_glsl_float(beta)} * input_c[{c_index}]"
    operation = (
        f"float sum = 0.0;\n"
        f"    for (uint inner = 0u; inner < {k}u; ++inner) {{\n"
        f"        sum += input_a[{a_index}] * input_b[{b_index}];\n"
        f"    }}\n"
        f"    output_y[index] = {_glsl_float(alpha)} * sum{c_term};"
    )
    source = _shader_source(
        tuple(declarations),
        "uint row = index / " + str(n) + "u;\n"
        "    uint column = index % " + str(n) + "u;\n"
        f"    bool trans_a = {str(trans_a).lower()};\n"
        f"    bool trans_b = {str(trans_b).lower()};\n"
        + operation,
    )
    return _single_dispatch_plan(
        context,
        source,
        m * n,
        input_count=len(context.inputs),
        kernel_id="linear.gemm.fp32",
    )


def _require_fp32(context: KernelContext) -> None:
    if any(tensor.data_type != "FLOAT" for tensor in (*context.inputs, *context.outputs)):
        types = ", ".join(tensor.data_type for tensor in (*context.inputs, *context.outputs))
        raise UnsupportedKernel(f"当前 Kernel 仅支持 FP32，当前类型为 {types}")
    if any(tensor.layout != "contiguous" for tensor in (*context.inputs, *context.outputs)):
        raise UnsupportedKernel("当前 Kernel 仅支持 contiguous layout")


def _attribute_flag(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise UnsupportedKernel(f"{name} 必须是 0 或 1")


def _finite_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise UnsupportedKernel(f"{name} 必须是有限浮点数")
    return float(value)


def _gemm_c_index(shape: tuple[int, ...], m: int, n: int) -> str:
    if shape in {(), (1,), (1, 1)}:
        return "0u"
    if shape in {(n,), (1, n)}:
        return "column"
    if shape == (m, 1):
        return "row"
    if shape == (m, n):
        return f"row * {n}u + column"
    raise UnsupportedKernel(
        f"Gemm C 形状 {shape} 不能广播到输出 ({m}, {n})"
    )


def _metadata_view(context: KernelContext) -> DispatchPlan:
    if context.op_type == "Identity":
        _require_arity(context, 1, 1)
        source = context.inputs[0]
        output = context.outputs[0]
        _require_known_shapes((source, output))
        if source.data_type == "UNKNOWN" or output.data_type == "UNKNOWN":
            raise UnsupportedKernel("Identity 缺少可验证的张量数据类型")
        if source.data_type != output.data_type or source.shape != output.shape:
            raise UnsupportedKernel("Identity 输入输出规格必须一致")
    elif context.op_type == "Reshape":
        _require_arity(context, 2, 1)
        source = context.inputs[0]
        output = context.outputs[0]
        _require_known_shapes((source, output))
        if source.data_type == "UNKNOWN" or output.data_type == "UNKNOWN":
            raise UnsupportedKernel("Reshape 缺少可验证的张量数据类型")
        if source.data_type != output.data_type:
            raise UnsupportedKernel("Reshape 不能改变张量数据类型")
        if _static_element_count(source.shape) != _static_element_count(output.shape):
            raise UnsupportedKernel("Reshape 输入输出元素数量必须一致")
    return DispatchPlan(
        kernel_id=f"view.{context.op_type.lower()}",
        operator=context.op_type,
        steps=(),
        metadata_only=True,
        notes=("该算子只更新张量视图，不提交 Vulkan Dispatch",),
    )


def _single_dispatch_plan(
    context: KernelContext,
    source: str,
    element_count: int,
    *,
    input_count: int,
    kernel_id: str | None = None,
    push_constants: dict[str, int | float] | None = None,
) -> DispatchPlan:
    workgroups = _linear_workgroups(element_count)
    bound_inputs = context.runtime_inputs or context.inputs
    bindings = tuple(
        BufferBinding(index, tensor.name or f"input_{index}", "read", tensor.data_type)
        for index, tensor in enumerate(bound_inputs)
    ) + (
        BufferBinding(
            len(bound_inputs),
            context.outputs[0].name or "output",
            "write",
            context.outputs[0].data_type,
        ),
    )
    if kernel_id is None:
        kernel_id = f"elementwise.{context.op_type.lower()}.fp32"
        if context.op_type == "Cast":
            kernel_id = "cast.fp32-int32"
    return DispatchPlan(
        kernel_id=kernel_id,
        operator=context.op_type,
        steps=(
            DispatchStep(
                shader=ShaderModule(kernel_id, source),
                workgroups=workgroups,
                bindings=bindings,
                push_constants={
                    "element_count": element_count,
                    **(push_constants or {}),
                },
            ),
        ),
    )


def _require_arity(context: KernelContext, inputs: int, outputs: int) -> None:
    if len(context.inputs) != inputs or len(context.outputs) != outputs:
        raise UnsupportedKernel(
            f"期望 {inputs} 个输入/{outputs} 个输出，实际为 "
            f"{len(context.inputs)}/{len(context.outputs)}"
        )


def _require_known_shapes(tensors: tuple[KernelTensor, ...]) -> None:
    unknown = [tensor.name or "(unnamed)" for tensor in tensors if not tensor.shape_known]
    if unknown:
        raise UnsupportedKernel(f"缺少可验证的张量秩: {', '.join(unknown)}")


def _require_same_static_fp32_shape(context: KernelContext) -> int:
    tensors = (*context.inputs, *context.outputs)
    if any(tensor.data_type != "FLOAT" for tensor in tensors):
        types = ", ".join(tensor.data_type for tensor in tensors)
        raise UnsupportedKernel(f"首版仅支持 FP32，当前类型为 {types}")
    shape = context.outputs[0].shape
    if any(tensor.shape != shape for tensor in context.inputs):
        raise UnsupportedKernel("首版暂不支持广播或不同输入形状")
    return _static_element_count(shape)


def _require_broadcast_static_fp32_shape(
    context: KernelContext,
) -> tuple[int, tuple[str, ...]]:
    tensors = (*context.inputs, *context.outputs)
    if any(tensor.data_type != "FLOAT" for tensor in tensors):
        types = ", ".join(tensor.data_type for tensor in tensors)
        raise UnsupportedKernel(f"首版仅支持 FP32，当前类型为 {types}")

    input_shapes = tuple(tuple(_static_shape(tensor.shape)) for tensor in context.inputs)
    output_shape = tuple(_static_shape(context.outputs[0].shape))
    inferred_shape = _broadcast_shape(input_shapes)
    if inferred_shape != output_shape:
        raise UnsupportedKernel(
            f"广播输出形状不匹配: 推导为 {inferred_shape}，实际为 {output_shape}"
        )
    element_count = _static_element_count(context.outputs[0].shape)
    input_indices = tuple(
        _broadcast_input_index(input_shape, output_shape)
        for input_shape in input_shapes
    )
    return element_count, input_indices


def _static_shape(shape: tuple[str, ...]) -> tuple[int, ...]:
    _static_element_count(shape)
    return tuple(int(dimension) for dimension in shape)


def _broadcast_shape(shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    rank = max((len(shape) for shape in shapes), default=0)
    result: list[int] = []
    for output_axis in range(rank):
        dimensions = [
            shape[len(shape) - rank + output_axis]
            for shape in shapes
            if output_axis >= rank - len(shape)
        ]
        non_unit = {dimension for dimension in dimensions if dimension != 1}
        if len(non_unit) > 1:
            raise UnsupportedKernel(f"输入形状无法广播: {shapes}")
        result.append(next(iter(non_unit), 1))
    return tuple(result)


def _broadcast_input_index(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> str:
    if not input_shape:
        return "0u"
    if input_shape == output_shape:
        return "index"

    rank_offset = len(output_shape) - len(input_shape)
    input_strides = _row_major_strides(input_shape)
    output_strides = _row_major_strides(output_shape)
    terms: list[str] = []
    for input_axis, (dimension, input_stride) in enumerate(
        zip(input_shape, input_strides, strict=True)
    ):
        if dimension == 1:
            continue
        output_axis = rank_offset + input_axis
        output_dimension = output_shape[output_axis]
        output_stride = output_strides[output_axis]
        if output_stride == 1:
            coordinate = f"(index % {output_dimension}u)"
        else:
            coordinate = (
                f"((index / {output_stride}u) % {output_dimension}u)"
            )
        if input_stride != 1:
            coordinate += f" * {input_stride}u"
        terms.append(coordinate)
    return " + ".join(terms) if terms else "0u"


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _transpose_permutation(value: object, rank: int) -> tuple[int, ...]:
    if value is None:
        return tuple(reversed(range(rank)))
    if not isinstance(value, (list, tuple)) or any(
        isinstance(axis, bool) or not isinstance(axis, int) for axis in value
    ):
        raise UnsupportedKernel("Transpose perm 必须是整数数组")
    permutation = tuple(value)
    if len(permutation) != rank or sorted(permutation) != list(range(rank)):
        raise UnsupportedKernel(
            f"Transpose perm 不是秩 {rank} 的有效排列: {permutation}"
        )
    return permutation


def _normalize_axis(value: object, rank: int, operator: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsupportedKernel(f"{operator} axis 必须是整数")
    axis = value + rank if value < 0 else value
    if rank == 0 or axis < 0 or axis >= rank:
        raise UnsupportedKernel(f"{operator} axis {value} 超出秩 {rank}")
    return axis


def _glsl_float(value: float) -> str:
    literal = f"{value:.9g}"
    return literal if "." in literal or "e" in literal.lower() else f"{literal}.0"


def _transpose_input_index(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    permutation: tuple[int, ...],
) -> str:
    if not input_shape or permutation == tuple(range(len(input_shape))):
        return "index"
    input_strides = _row_major_strides(input_shape)
    output_strides = _row_major_strides(output_shape)
    terms: list[str] = []
    for output_axis, input_axis in enumerate(permutation):
        output_dimension = output_shape[output_axis]
        if output_dimension == 1:
            continue
        output_stride = output_strides[output_axis]
        if output_stride == 1:
            coordinate = f"(index % {output_dimension}u)"
        else:
            coordinate = f"((index / {output_stride}u) % {output_dimension}u)"
        input_stride = input_strides[input_axis]
        if input_stride != 1:
            coordinate += f" * {input_stride}u"
        terms.append(coordinate)
    return " + ".join(terms) if terms else "0u"


def _static_element_count(shape: tuple[str, ...]) -> int:
    element_count = 1
    for dimension in shape:
        if not dimension.isdigit():
            raise UnsupportedKernel(f"首版要求静态形状，发现维度 {dimension or 'UNKNOWN'}")
        element_count *= int(dimension)
    if element_count <= 0:
        raise UnsupportedKernel("首版要求非空张量")
    if element_count > MAX_SHADER_ELEMENTS:
        raise UnsupportedKernel(
            f"张量超过当前 uint 索引上限 {MAX_SHADER_ELEMENTS}: {element_count}"
        )
    return element_count


def _linear_workgroups(element_count: int) -> tuple[int, int, int]:
    group_count = (element_count + 255) // 256
    groups_x = min(group_count, MAX_DISPATCH_GROUP_COUNT)
    groups_y = (group_count + groups_x - 1) // groups_x
    if groups_y > MAX_DISPATCH_GROUP_COUNT:
        raise UnsupportedKernel(
            f"张量需要 {group_count} 个工作组，超过当前二维 dispatch 上限"
        )
    return groups_x, groups_y, 1


def _shader_source(bindings: tuple[str, ...], operation: str) -> str:
    declarations = "\n".join(bindings)
    return f"""#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
{declarations}
layout(push_constant) uniform Parameters {{ uint element_count; }} params;

void main() {{
    uint index = gl_GlobalInvocationID.x
        + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    if (index >= params.element_count) return;
    {operation}
}}
"""
