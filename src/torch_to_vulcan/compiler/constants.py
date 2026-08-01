"""Conservative ONNX compile-time evaluation and package materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
    numpy_helper,
)
from onnx.reference import ReferenceEvaluator

from .package import ExecutablePackageBuilder, ExecutablePackageError

DEFAULT_FOLDABLE_OPERATORS = frozenset(
    {
        "Constant",
        "Shape",
        "Size",
        "Cast",
        "Identity",
        "Unsqueeze",
        "Squeeze",
        "Reshape",
        "Slice",
        "Gather",
        "Concat",
        "Add",
        "Sub",
        "Mul",
        "Div",
    }
)
DEFAULT_MAX_CONSTANT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FoldedNode:
    index: int
    name: str
    op_type: str
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FoldDiagnostic:
    index: int
    name: str
    op_type: str
    message: str


@dataclass(slots=True)
class ConstantFoldResult:
    values: dict[str, np.ndarray] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    folded_nodes: list[FoldedNode] = field(default_factory=list)
    runtime_node_indices: list[int] = field(default_factory=list)
    required_values: list[str] = field(default_factory=list)
    diagnostics: list[FoldDiagnostic] = field(default_factory=list)

    @property
    def folded_node_indices(self) -> frozenset[int]:
        return frozenset(node.index for node in self.folded_nodes)


def fold_constant_subgraph(
    model: ModelProto,
    graph: GraphProto | None = None,
    *,
    foldable_operators: Iterable[str] = DEFAULT_FOLDABLE_OPERATORS,
    max_constant_bytes: int = DEFAULT_MAX_CONSTANT_BYTES,
) -> ConstantFoldResult:
    """Evaluate a topologically ordered standard-domain subgraph when inputs are known."""
    selected_graph = graph or model.graph
    allowed = frozenset(foldable_operators)
    result = ConstantFoldResult()
    for initializer in selected_graph.initializer:
        if not initializer.name or initializer.data_location == TensorProto.EXTERNAL:
            continue
        try:
            array = _safe_array(
                numpy_helper.to_array(initializer),
                max_constant_bytes,
            )
        except (ExecutablePackageError, OSError, TypeError, ValueError) as error:
            result.diagnostics.append(
                FoldDiagnostic(-1, initializer.name, "Initializer", str(error))
            )
            continue
        result.values[initializer.name] = array
        result.sources[initializer.name] = "initializer"

    static_shapes = _static_tensor_shapes(selected_graph)
    opsets = {item.domain: item.version for item in model.opset_import}
    for index, node in enumerate(selected_graph.node):
        if node.domain or node.op_type not in allowed:
            result.runtime_node_indices.append(index)
            continue
        if node.op_type in {"Shape", "Size"} and node.input and node.input[0] not in result.values:
            shape = static_shapes.get(node.input[0])
            if shape is None:
                result.runtime_node_indices.append(index)
                continue
            outputs = (
                _evaluate_static_shape(node, shape)
                if node.op_type == "Shape"
                else (np.asarray(np.prod(shape), dtype=np.int64),)
            )
        elif all(not name or name in result.values for name in node.input):
            try:
                outputs = _evaluate_node(node, result.values, opsets)
            except Exception as error:
                result.diagnostics.append(
                    FoldDiagnostic(
                        index,
                        node.name,
                        node.op_type,
                        str(error) or type(error).__name__,
                    )
                )
                result.runtime_node_indices.append(index)
                continue
        else:
            result.runtime_node_indices.append(index)
            continue

        try:
            arrays = tuple(
                _safe_array(value, max_constant_bytes) for value in outputs
            )
        except (ExecutablePackageError, TypeError, ValueError) as error:
            result.diagnostics.append(
                FoldDiagnostic(index, node.name, node.op_type, str(error))
            )
            result.runtime_node_indices.append(index)
            continue
        output_names = tuple(name for name in node.output if name)
        if len(arrays) != len(output_names):
            result.diagnostics.append(
                FoldDiagnostic(
                    index,
                    node.name,
                    node.op_type,
                    f"output count {len(arrays)} != {len(output_names)}",
                )
            )
            result.runtime_node_indices.append(index)
            continue
        for name, array in zip(output_names, arrays, strict=True):
            result.values[name] = array
            result.sources[name] = f"node:{index}"
        result.folded_nodes.append(
            FoldedNode(index, node.name, node.op_type, output_names)
        )
    result.required_values.extend(
        _required_constant_values(selected_graph, result)
    )
    return result


def materialize_folded_constants(
    builder: ExecutablePackageBuilder,
    result: ConstantFoldResult,
    *,
    names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Append selected folded values to a package constant blob."""
    selected = tuple(names) if names is not None else tuple(result.required_values)
    for name in selected:
        if name not in result.values:
            raise ExecutablePackageError(f"unknown folded constant {name}")
        array = _contiguous_array(result.values[name])
        data_type = _onnx_data_type_name(array.dtype)
        builder.add_constant(
            name,
            data_type,
            array.shape,
            array.tobytes(order="C"),
        )
    return selected


def rewrite_model_with_folded_constants(
    model: ModelProto,
    result: ConstantFoldResult,
) -> ModelProto:
    """Return a root-graph model with folded nodes replaced by boundary initializers."""
    rewritten = ModelProto()
    rewritten.CopyFrom(model)
    graph = rewritten.graph
    folded_indices = result.folded_node_indices
    runtime_nodes = [
        node for index, node in enumerate(graph.node) if index not in folded_indices
    ]
    del graph.node[:]
    graph.node.extend(runtime_nodes)

    retained_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in result.values
    ]
    for name in result.required_values:
        array = _contiguous_array(result.values[name])
        retained_initializers.append(numpy_helper.from_array(array, name=name))
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    return rewritten


def _evaluate_node(
    node: NodeProto,
    values: Mapping[str, np.ndarray],
    opsets: Mapping[str, int],
) -> Sequence[object]:
    if node.op_type == "Constant":
        return (_constant_node_value(node),)
    feeds = {name: values[name] for name in node.input if name}
    evaluator = ReferenceEvaluator(node, opsets=dict(opsets))
    return evaluator.run(None, feeds)


def _constant_node_value(node: NodeProto) -> np.ndarray:
    attributes = {attribute.name: attribute for attribute in node.attribute}
    if "value" in attributes and attributes["value"].type == AttributeProto.TENSOR:
        return numpy_helper.to_array(attributes["value"].t)
    scalar_values = {
        "value_float": (AttributeProto.FLOAT, np.float32, "f"),
        "value_int": (AttributeProto.INT, np.int64, "i"),
        "value_string": (AttributeProto.STRING, np.bytes_, "s"),
    }
    for name, (attribute_type, dtype, field_name) in scalar_values.items():
        attribute = attributes.get(name)
        if attribute is not None and attribute.type == attribute_type:
            return np.asarray(getattr(attribute, field_name), dtype=dtype)
    vector_values = {
        "value_floats": (AttributeProto.FLOATS, np.float32, "floats"),
        "value_ints": (AttributeProto.INTS, np.int64, "ints"),
        "value_strings": (AttributeProto.STRINGS, np.bytes_, "strings"),
    }
    for name, (attribute_type, dtype, field_name) in vector_values.items():
        attribute = attributes.get(name)
        if attribute is not None and attribute.type == attribute_type:
            return np.asarray(list(getattr(attribute, field_name)), dtype=dtype)
    raise ExecutablePackageError("Constant has no supported value attribute")


def _evaluate_static_shape(node: NodeProto, shape: tuple[int, ...]) -> tuple[np.ndarray]:
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    rank = len(shape)
    start = int(attributes.get("start", 0))
    end = int(attributes.get("end", rank))
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    start = min(max(start, 0), rank)
    end = min(max(end, 0), rank)
    return (np.asarray(shape[start:end], dtype=np.int64),)


def _static_tensor_shapes(graph: GraphProto) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {
        initializer.name: tuple(initializer.dims)
        for initializer in graph.initializer
        if initializer.name
    }
    for value in (*graph.input, *graph.output, *graph.value_info):
        if not value.name or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions: list[int] = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value < 0:
                break
            dimensions.append(dimension.dim_value)
        else:
            result[value.name] = tuple(dimensions)
    return result


def _required_constant_values(
    graph: GraphProto,
    result: ConstantFoldResult,
) -> tuple[str, ...]:
    required: list[str] = []
    seen: set[str] = set()
    for node_index in result.runtime_node_indices:
        for name in graph.node[node_index].input:
            if name in result.values and name not in seen:
                required.append(name)
                seen.add(name)
    for output in graph.output:
        if output.name in result.values and output.name not in seen:
            required.append(output.name)
            seen.add(output.name)
    return tuple(required)


def _safe_array(value: object, max_constant_bytes: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "V"}:
        raise ExecutablePackageError(f"unsupported compile-time dtype {array.dtype}")
    if array.nbytes > max_constant_bytes:
        raise ExecutablePackageError(
            f"compile-time value is {array.nbytes} bytes; limit is {max_constant_bytes}"
        )
    return _contiguous_array(array)


def _contiguous_array(array: np.ndarray) -> np.ndarray:
    # np.ascontiguousarray promotes a scalar from rank 0 to rank 1.
    return array.copy() if array.ndim == 0 else np.ascontiguousarray(array)


def _onnx_data_type_name(dtype: np.dtype) -> str:
    try:
        data_type = helper.np_dtype_to_tensor_dtype(dtype)
        return TensorProto.DataType.Name(data_type)
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutablePackageError(f"unsupported package constant dtype {dtype}") from error
