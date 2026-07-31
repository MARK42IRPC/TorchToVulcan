"""Serializable result types produced by ONNX source inspection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class OpsetReport:
    domain: str
    version: int


@dataclass(frozen=True, slots=True)
class OperatorReport:
    graph_path: str
    index: int
    name: str
    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TensorValueReport:
    name: str
    data_type: str
    shape: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphReport:
    path: str
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    values: tuple[TensorValueReport, ...]
    operators: tuple[OperatorReport, ...]


@dataclass(frozen=True, slots=True)
class ModelReport:
    path: str
    graph_name: str
    ir_version: int
    producer_name: str
    producer_version: str
    opsets: tuple[OpsetReport, ...]
    graphs: tuple[GraphReport, ...]

    @property
    def operator_count(self) -> int:
        return sum(len(graph.operators) for graph in self.graphs)


@dataclass(frozen=True, slots=True)
class ModelError:
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class OperatorCount:
    domain: str
    op_type: str
    count: int


@dataclass(slots=True)
class InspectionReport:
    source: str
    source_type: str
    models: list[ModelReport] = field(default_factory=list)
    errors: list[ModelError] = field(default_factory=list)

    @property
    def operator_count(self) -> int:
        return sum(model.operator_count for model in self.models)

    @property
    def operator_summary(self) -> list[OperatorCount]:
        counts: Counter[tuple[str, str]] = Counter()
        for model in self.models:
            for graph in model.graphs:
                counts.update((operator.domain, operator.op_type) for operator in graph.operators)
        return [
            OperatorCount(domain=domain, op_type=op_type, count=count)
            for (domain, op_type), count in sorted(counts.items())
        ]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for model_value, model in zip(result["models"], self.models, strict=True):
            model_value["operator_count"] = model.operator_count
        result["operator_count"] = self.operator_count
        result["operator_summary"] = [asdict(item) for item in self.operator_summary]
        return result


def display_domain(domain: str) -> str:
    """Return the conventional name for the default ONNX domain."""

    return domain or "ai.onnx"
