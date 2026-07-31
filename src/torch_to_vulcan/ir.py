"""Version 1.0 semantic Graph IR.

The JSON Schema in ``schemas/graph-ir.schema.json`` is the language-neutral
contract. These dataclasses provide the first compiler-side representation and
enforce invariants that JSON Schema cannot express across objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias


SCHEMA_VERSION = "1.0"
Dimension: TypeAlias = int | str | None
AttributeValue: TypeAlias = None | bool | int | float | str | list["AttributeValue"]


class DataType(StrEnum):
    BOOL = "bool"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    INT64 = "int64"
    UINT64 = "uint64"


@dataclass(slots=True)
class ModelInfo:
    name: str
    producer_name: str = ""
    producer_version: str = ""
    source_format: str = "onnx"
    source_ir_version: int | None = None


@dataclass(slots=True)
class Opset:
    domain: str
    version: int


@dataclass(slots=True)
class DataRef:
    uri: str
    offset: int
    length: int
    sha256: str | None = None


@dataclass(slots=True)
class Quantization:
    scheme: str
    scale: float | list[float]
    zero_point: int | list[int]
    axis: int | None = None


@dataclass(slots=True)
class Tensor:
    dtype: DataType
    shape: list[Dimension]
    layout: str = "UNKNOWN"
    data: DataRef | None = None
    quantization: Quantization | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def is_constant(self) -> bool:
        return self.data is not None


@dataclass(slots=True)
class Node:
    id: str
    op_type: str
    inputs: list[str | None]
    outputs: list[str]
    domain: str = ""
    name: str = ""
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NodeUiState:
    x: float
    y: float
    collapsed: bool = False


@dataclass(slots=True)
class Viewport:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


@dataclass(slots=True)
class UiState:
    nodes: dict[str, NodeUiState] = field(default_factory=dict)
    viewport: Viewport | None = None


class GraphValidationError(ValueError):
    """Raised when a graph violates a cross-object Graph IR invariant."""


@dataclass(slots=True)
class Graph:
    model: ModelInfo
    opsets: list[Opset]
    inputs: list[str]
    outputs: list[str]
    tensors: dict[str, Tensor]
    nodes: list[Node]
    schema_version: str = SCHEMA_VERSION
    ui: UiState | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        errors: list[str] = []

        if self.schema_version != SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not self.model.name:
            errors.append("model.name must not be empty")

        tensor_ids = set(self.tensors)
        self._validate_unique("graph input", self.inputs, errors)
        self._validate_unique("graph output", self.outputs, errors)

        for tensor_id in [*self.inputs, *self.outputs]:
            if tensor_id not in tensor_ids:
                errors.append(f"graph references unknown tensor {tensor_id!r}")

        available = set(self.inputs)
        available.update(
            tensor_id for tensor_id, tensor in self.tensors.items() if tensor.is_constant
        )
        node_ids: set[str] = set()
        produced: set[str] = set()

        for index, node in enumerate(self.nodes):
            label = f"nodes[{index}] ({node.id!r})"
            if not node.id:
                errors.append(f"{label} has an empty id")
            elif node.id in node_ids:
                errors.append(f"duplicate node id {node.id!r}")
            node_ids.add(node.id)

            if not node.op_type:
                errors.append(f"{label} has an empty op_type")
            if not node.outputs:
                errors.append(f"{label} has no outputs")

            for tensor_id in node.inputs:
                if tensor_id is None:
                    continue
                if tensor_id not in tensor_ids:
                    errors.append(f"{label} input references unknown tensor {tensor_id!r}")
                elif tensor_id not in available:
                    errors.append(
                        f"{label} input {tensor_id!r} is unavailable; nodes must be topologically ordered"
                    )

            for tensor_id in node.outputs:
                if tensor_id not in tensor_ids:
                    errors.append(f"{label} output references unknown tensor {tensor_id!r}")
                if tensor_id in produced:
                    errors.append(f"tensor {tensor_id!r} has more than one producer")
                if tensor_id in self.inputs:
                    errors.append(f"{label} overwrites graph input {tensor_id!r}")
                produced.add(tensor_id)
                available.add(tensor_id)

        for tensor_id in self.outputs:
            if tensor_id not in available:
                errors.append(f"graph output {tensor_id!r} is never produced")

        if self.ui is not None:
            unknown_ui_nodes = set(self.ui.nodes) - node_ids
            for node_id in sorted(unknown_ui_nodes):
                errors.append(f"ui state references unknown node {node_id!r}")
            if self.ui.viewport is not None and self.ui.viewport.zoom <= 0:
                errors.append("ui viewport zoom must be greater than zero")

        if errors:
            raise GraphValidationError("invalid Graph IR:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary, omitting absent optional values."""

        return _without_none(asdict(self))

    @staticmethod
    def _validate_unique(label: str, values: list[str], errors: list[str]) -> None:
        seen: set[str] = set()
        for value in values:
            if value in seen:
                errors.append(f"duplicate {label} {value!r}")
            seen.add(value)


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_none(item)
            for key, item in value.items()
            if item is not None and not (key == "metadata" and item == {})
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def graph_from_dict(value: dict[str, Any]) -> Graph:
    """Decode a JSON-compatible dictionary and validate its graph invariants."""

    tensors = {
        tensor_id: Tensor(
            dtype=DataType(tensor["dtype"]),
            shape=list(tensor["shape"]),
            layout=tensor["layout"],
            data=DataRef(**tensor["data"]) if "data" in tensor else None,
            quantization=(
                Quantization(**tensor["quantization"]) if "quantization" in tensor else None
            ),
            metadata=dict(tensor.get("metadata", {})),
        )
        for tensor_id, tensor in value["tensors"].items()
    }
    nodes = [Node(**node) for node in value["nodes"]]

    ui_value = value.get("ui")
    ui = None
    if ui_value is not None:
        ui = UiState(
            nodes={
                node_id: NodeUiState(**node_ui)
                for node_id, node_ui in ui_value.get("nodes", {}).items()
            },
            viewport=(Viewport(**ui_value["viewport"]) if "viewport" in ui_value else None),
        )

    graph = Graph(
        schema_version=value.get("schema_version", SCHEMA_VERSION),
        model=ModelInfo(**value["model"]),
        opsets=[Opset(**opset) for opset in value["opsets"]],
        inputs=list(value["inputs"]),
        outputs=list(value["outputs"]),
        tensors=tensors,
        nodes=nodes,
        ui=ui,
        metadata=dict(value.get("metadata", {})),
    )
    graph.validate()
    return graph
