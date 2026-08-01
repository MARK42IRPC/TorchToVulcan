"""Normalized ONNX IR used by compiler passes and backend selection.

The ONNX protobuf is deliberately kept at the importer boundary.  Compiler
passes consume these immutable records instead, which gives shape, layout,
stride, attributes, and nested graphs one stable representation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from onnx import GraphProto, ModelProto, TensorProto, helper, numpy_helper

from ..contracts import ContractError, ShapeProfile

Dimension = int | str | None
Shape = tuple[Dimension, ...] | None


class NormalizationError(ValueError):
    """The source model cannot be represented by the normalized ONNX IR."""


@dataclass(frozen=True, slots=True)
class NormalizedTensor:
    """Tensor type and memory contract known to compiler passes."""

    name: str
    data_type: str
    shape: Shape
    layout: str = "contiguous"
    strides: tuple[int, ...] | None = None
    source: str = "value_info"

    def __post_init__(self) -> None:
        if not self.name:
            raise NormalizationError("normalized tensor name must not be empty")
        if not self.data_type:
            raise NormalizationError(f"tensor {self.name!r} data type must not be empty")
        if not self.layout:
            raise NormalizationError(f"tensor {self.name!r} layout must not be empty")
        if self.shape is not None:
            for dimension in self.shape:
                if isinstance(dimension, bool):
                    raise NormalizationError(
                        f"tensor {self.name!r} dimensions cannot be boolean"
                    )
                if isinstance(dimension, int) and dimension < 0:
                    raise NormalizationError(
                        f"tensor {self.name!r} dimensions must be non-negative"
                    )
                if dimension is not None and not isinstance(dimension, (int, str)):
                    raise NormalizationError(
                        f"tensor {self.name!r} has an invalid dimension {dimension!r}"
                    )
        if self.strides is not None:
            if self.shape is None or len(self.strides) != len(self.shape):
                raise NormalizationError(
                    f"tensor {self.name!r} strides must match a known tensor rank"
                )
            if any(
                isinstance(stride, bool) or not isinstance(stride, int) or stride < 0
                for stride in self.strides
            ):
                raise NormalizationError(
                    f"tensor {self.name!r} strides must be non-negative integers"
                )

    @property
    def rank_known(self) -> bool:
        return self.shape is not None

    @property
    def fully_static(self) -> bool:
        return self.shape is not None and all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in self.shape
        )

    @property
    def static_shape(self) -> tuple[int, ...]:
        if not self.fully_static:
            raise NormalizationError(f"tensor {self.name!r} does not have a static shape")
        return tuple(self.shape or ())  # type: ignore[arg-type]

    @property
    def element_count(self) -> int:
        return int(np.prod(self.static_shape, dtype=np.int64))

    def specialize(self, profile: ShapeProfile) -> "NormalizedTensor":
        if self.shape is None:
            return self
        try:
            resolved = tuple(profile.resolve_dimension(dimension) for dimension in self.shape)
        except ContractError as error:
            raise NormalizationError(
                f"tensor {self.name!r} cannot be specialized by profile "
                f"{profile.name!r}: {error}"
            ) from error
        return replace(
            self,
            shape=resolved,
            strides=_row_major_strides(resolved),
        )


@dataclass(frozen=True, slots=True)
class NormalizedAttributeTensor:
    """Tensor-valued node attribute without retaining an ONNX protobuf."""

    data_type: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class NormalizedConstant:
    tensor: NormalizedTensor
    data: bytes


@dataclass(frozen=True, slots=True)
class NormalizedNode:
    index: int
    name: str
    domain: str
    op_type: str
    opset_version: int
    inputs: tuple[str | None, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any]
    subgraphs: tuple["NormalizedGraph", ...] = ()

    def __post_init__(self) -> None:
        if self.index < 0:
            raise NormalizationError("normalized node index must be non-negative")
        if not self.op_type:
            raise NormalizationError("normalized node op_type must not be empty")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def has_subgraphs(self) -> bool:
        return bool(self.subgraphs)


@dataclass(frozen=True, slots=True)
class NormalizedGraph:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tensors: Mapping[str, NormalizedTensor]
    constants: Mapping[str, NormalizedConstant]
    nodes: tuple[NormalizedNode, ...]
    opsets: Mapping[str, int]
    captures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(self, "constants", MappingProxyType(dict(self.constants)))
        object.__setattr__(self, "opsets", MappingProxyType(dict(self.opsets)))
        object.__setattr__(self, "captures", tuple(self.captures))
        self.validate()

    def validate(self) -> None:
        errors: list[str] = []
        tensor_ids = set(self.tensors)
        if len(set(self.inputs)) != len(self.inputs):
            errors.append("graph inputs must be unique")
        if len(set(self.outputs)) != len(self.outputs):
            errors.append("graph outputs must be unique")
        if len(set(self.captures)) != len(self.captures):
            errors.append("graph captures must be unique")
        if set(self.captures) & set(self.inputs):
            errors.append("graph captures must not shadow graph inputs")
        for tensor_id in (*self.inputs, *self.outputs):
            if tensor_id not in tensor_ids:
                errors.append(f"graph references unknown tensor {tensor_id!r}")
        for tensor_id in self.captures:
            if tensor_id not in tensor_ids:
                errors.append(f"graph capture references unknown tensor {tensor_id!r}")
        for tensor_id, constant in self.constants.items():
            if tensor_id not in tensor_ids:
                errors.append(f"constant {tensor_id!r} has no tensor record")
            elif constant.tensor.name != tensor_id:
                errors.append(f"constant {tensor_id!r} does not match its tensor name")

        available = set(self.inputs) | set(self.constants) | set(self.captures)
        produced: set[str] = set()
        for node in self.nodes:
            for tensor_id in node.inputs:
                if tensor_id is not None and tensor_id not in tensor_ids:
                    errors.append(
                        f"node {node.index} input references unknown tensor {tensor_id!r}"
                    )
                if tensor_id is not None and tensor_id not in available:
                    errors.append(
                        f"node {node.index} input {tensor_id!r} is unavailable; "
                        "normalized nodes must be topologically ordered"
                    )
            for tensor_id in node.outputs:
                if tensor_id not in tensor_ids:
                    errors.append(
                        f"node {node.index} output references unknown tensor {tensor_id!r}"
                    )
                if tensor_id in produced:
                    errors.append(f"tensor {tensor_id!r} has more than one producer")
                if tensor_id in self.inputs:
                    errors.append(f"node {node.index} overwrites graph input {tensor_id!r}")
                produced.add(tensor_id)
                available.add(tensor_id)
        for tensor_id in self.outputs:
            if tensor_id not in available:
                errors.append(f"graph output {tensor_id!r} is never produced")
        if errors:
            raise NormalizationError("invalid normalized graph:\n- " + "\n- ".join(errors))

    def specialize(self, profile: ShapeProfile) -> "NormalizedGraph":
        tensors = {
            name: tensor.specialize(profile) for name, tensor in self.tensors.items()
        }
        constants = {
            name: replace(constant, tensor=tensors[name])
            for name, constant in self.constants.items()
        }
        nodes = tuple(
            replace(
                node,
                subgraphs=tuple(subgraph.specialize(profile) for subgraph in node.subgraphs),
            )
            for node in self.nodes
        )
        return NormalizedGraph(
            self.name,
            self.inputs,
            self.outputs,
            tensors,
            constants,
            nodes,
            self.opsets,
            self.captures,
        )


@dataclass(frozen=True, slots=True)
class NormalizedModel:
    graph: NormalizedGraph
    ir_version: int
    producer_name: str
    producer_version: str
    model_name: str

    def specialize(self, profile: ShapeProfile) -> "NormalizedModel":
        return replace(self, graph=self.graph.specialize(profile))


def normalize_onnx_model(model: ModelProto) -> NormalizedModel:
    """Convert an ONNX model into the compiler-owned normalized IR."""
    opsets = {
        _normalize_domain(item.domain): int(item.version) for item in model.opset_import
    }
    graph = _normalize_graph(model.graph, opsets)
    return NormalizedModel(
        graph=graph,
        ir_version=int(model.ir_version),
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        model_name=graph.name or "onnx-model",
    )


def _normalize_graph(
    graph: GraphProto,
    opsets: Mapping[str, int],
    outer_tensors: Mapping[str, NormalizedTensor] | None = None,
) -> NormalizedGraph:
    outer_tensors = outer_tensors or {}
    tensor_records: dict[str, NormalizedTensor] = {}
    constants: dict[str, NormalizedConstant] = {}

    for initializer in graph.initializer:
        if not initializer.name:
            continue
        tensor = _initializer_tensor(initializer)
        tensor_records[tensor.name] = tensor
        try:
            array = numpy_helper.to_array(initializer)
        except (TypeError, ValueError, RuntimeError) as error:
            raise NormalizationError(
                f"cannot materialize initializer {initializer.name!r}: {error}"
            ) from error
        contiguous = array.copy() if array.ndim == 0 else np.ascontiguousarray(array)
        constants[initializer.name] = NormalizedConstant(
            tensor=tensor,
            data=contiguous.tobytes(order="C"),
        )

    for value in (*graph.input, *graph.value_info, *graph.output):
        if not value.name or not value.type.HasField("tensor_type"):
            continue
        tensor = _value_info_tensor(value.name, value.type.tensor_type)
        previous = tensor_records.get(value.name)
        tensor_records[value.name] = _merge_tensor_records(previous, tensor)

    local_tensor_names = set(tensor_records)
    capture_names: list[str] = []
    for node in graph.node:
        for name in node.input:
            if (
                name
                and name not in local_tensor_names
                and name in outer_tensors
                and name not in capture_names
            ):
                capture_names.append(name)
    captures = tuple(capture_names)
    for name in captures:
        tensor_records.setdefault(name, replace(outer_tensors[name], source="capture"))

    nodes: list[NormalizedNode] = []
    for index, node in enumerate(graph.node):
        attributes: dict[str, Any] = {}
        subgraphs: list[NormalizedGraph] = []
        for attribute in node.attribute:
            value = helper.get_attribute_value(attribute)
            normalized = _normalize_attribute_value(
                value,
                opsets,
                subgraphs,
                tensor_records,
            )
            attributes[attribute.name] = normalized
        nodes.append(
            NormalizedNode(
                index=index,
                name=node.name,
                domain=_normalize_domain(node.domain),
                op_type=node.op_type,
                opset_version=opsets.get(_normalize_domain(node.domain), 0),
                inputs=tuple(name if name else None for name in node.input),
                outputs=tuple(name for name in node.output if name),
                attributes=attributes,
                subgraphs=tuple(subgraphs),
            )
        )

    return NormalizedGraph(
        name=graph.name,
        inputs=tuple(value.name for value in graph.input if value.name),
        outputs=tuple(value.name for value in graph.output if value.name),
        tensors=tensor_records,
        constants=constants,
        nodes=tuple(nodes),
        opsets=opsets,
        captures=captures,
    )


def _initializer_tensor(initializer: TensorProto) -> NormalizedTensor:
    return NormalizedTensor(
        name=initializer.name,
        data_type=_data_type_name(initializer.data_type),
        shape=tuple(int(dimension) for dimension in initializer.dims),
        strides=_row_major_strides(tuple(int(dimension) for dimension in initializer.dims)),
        source="initializer",
    )


def _value_info_tensor(name: str, tensor_type: TensorProto.TensorType) -> NormalizedTensor:
    shape: Shape = None
    if tensor_type.HasField("shape"):
        dimensions: list[Dimension] = []
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param") and dimension.dim_param:
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
        shape = tuple(dimensions)
    strides = (
        _row_major_strides(shape)  # type: ignore[arg-type]
        if shape is not None and all(isinstance(item, int) for item in shape)
        else None
    )
    return NormalizedTensor(
        name=name,
        data_type=_data_type_name(tensor_type.elem_type),
        shape=shape,
        strides=strides,
        source="value_info",
    )


def _merge_tensor_records(
    previous: NormalizedTensor | None,
    current: NormalizedTensor,
) -> NormalizedTensor:
    if previous is None:
        return current
    data_type = previous.data_type
    if data_type in {"UNDEFINED", "UNKNOWN"} and current.data_type not in {
        "UNDEFINED",
        "UNKNOWN",
    }:
        data_type = current.data_type
    shape = previous.shape if previous.shape is not None else current.shape
    if previous.shape is not None and current.shape is not None:
        shape = tuple(
            old if old is not None else new
            for old, new in zip(previous.shape, current.shape, strict=False)
        )
    strides = (
        _row_major_strides(shape)  # type: ignore[arg-type]
        if shape is not None and all(isinstance(item, int) for item in shape)
        else None
    )
    return replace(previous, data_type=data_type, shape=shape, strides=strides)


def _normalize_attribute_value(
    value: Any,
    opsets: Mapping[str, int],
    subgraphs: list[NormalizedGraph],
    outer_tensors: Mapping[str, NormalizedTensor],
) -> Any:
    if isinstance(value, GraphProto):
        normalized = _normalize_graph(value, opsets, outer_tensors)
        subgraphs.append(normalized)
        return normalized
    if isinstance(value, TensorProto):
        return _normalize_attribute_tensor(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return tuple(
            _normalize_attribute_value(item, opsets, subgraphs, outer_tensors)
            for item in value
        )
    if isinstance(value, tuple):
        return tuple(
            _normalize_attribute_value(item, opsets, subgraphs, outer_tensors)
            for item in value
        )
    return value


def _normalize_attribute_tensor(tensor: TensorProto) -> NormalizedAttributeTensor:
    array = numpy_helper.to_array(tensor)
    contiguous = array.copy() if array.ndim == 0 else np.ascontiguousarray(array)
    return NormalizedAttributeTensor(
        data_type=_data_type_name(tensor.data_type),
        shape=tuple(int(dimension) for dimension in tensor.dims),
        data=contiguous.tobytes(order="C"),
    )


def _data_type_name(value: int) -> str:
    try:
        return TensorProto.DataType.Name(int(value))
    except (TypeError, ValueError):
        return f"TYPE_{value}"


def _normalize_domain(domain: str) -> str:
    return "" if domain == "ai.onnx" else domain


def _row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(tuple(shape)):
        result.append(stride)
        stride *= int(dimension)
    return tuple(reversed(result))
