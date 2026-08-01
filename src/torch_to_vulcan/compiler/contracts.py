"""Stable contracts shared by ONNX normalization and backend compilers.

The first compiler slice still materializes static TTV 0.1 packages.  These
contracts deliberately describe the larger compiler boundary so that dynamic
profiles, layouts, and backend capability checks do not become ad-hoc fields
inside the ONNX adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

Dimension = int | str
Shape = tuple[Dimension, ...]


class ContractError(ValueError):
    """A compiler contract contains an invalid value."""


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    """Resolve symbolic ONNX dimensions for one concrete compilation profile."""

    name: str = "default"
    dimensions: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ContractError("shape profile name must not be empty")
        normalized: dict[str, int] = {}
        for symbol, value in dict(self.dimensions).items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ContractError("shape profile dimension names must not be empty")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(
                    f"shape profile dimension {symbol!r} must be a non-negative integer"
                )
            normalized[symbol] = value
        object.__setattr__(self, "dimensions", MappingProxyType(normalized))

    @classmethod
    def from_mapping(
        cls,
        dimensions: Mapping[str, int],
        *,
        name: str = "default",
    ) -> "ShapeProfile":
        return cls(name=name, dimensions=dimensions)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShapeProfile":
        """Parse the stable JSON representation used by the CLI and packages."""
        name = value.get("name", "default")
        dimensions = value.get("dimensions", {})
        if not isinstance(name, str) or not isinstance(dimensions, Mapping):
            raise ContractError("shape profile requires a string name and object dimensions")
        return cls.from_mapping(dimensions, name=name)

    def resolve_dimension(self, dimension: Dimension) -> int:
        if isinstance(dimension, bool):
            raise ContractError("shape dimensions cannot be boolean")
        if isinstance(dimension, int):
            if dimension < 0:
                raise ContractError("shape dimensions must be non-negative")
            return dimension
        try:
            return self.dimensions[dimension]
        except KeyError as error:
            raise ContractError(
                f"shape profile {self.name!r} does not bind dimension {dimension!r}"
            ) from error

    def resolve_shape(self, shape: Shape) -> tuple[int, ...]:
        return tuple(self.resolve_dimension(dimension) for dimension in shape)

    def unresolved(self, shape: Shape) -> tuple[str, ...]:
        return tuple(
            dimension
            for dimension in shape
            if isinstance(dimension, str) and dimension not in self.dimensions
        )

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "dimensions": dict(self.dimensions)}


@dataclass(frozen=True, slots=True)
class OperatorCapability:
    """One backend kernel candidate's declared support boundary."""

    domain: str
    op_type: str
    min_opset: int = 1
    max_opset: int | None = None
    data_types: frozenset[str] = frozenset({"FLOAT"})
    layouts: frozenset[str] = frozenset({"contiguous"})
    control_flow: str = "none"
    notes: tuple[str, ...] = ()

    def matches(
        self,
        *,
        domain: str,
        op_type: str,
        opset_version: int,
        data_types: Sequence[str],
        layout: str,
    ) -> bool:
        if self.domain != domain or self.op_type != op_type:
            return False
        if opset_version < self.min_opset:
            return False
        if self.max_opset is not None and opset_version > self.max_opset:
            return False
        if layout not in self.layouts and "*" not in self.layouts:
            return False
        if "*" not in self.data_types and any(
            data_type not in self.data_types for data_type in data_types
        ):
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "op_type": self.op_type,
            "min_opset": self.min_opset,
            "max_opset": self.max_opset,
            "data_types": sorted(self.data_types),
            "layouts": sorted(self.layouts),
            "control_flow": self.control_flow,
            "notes": list(self.notes),
        }


class CapabilitySource(Protocol):
    def capabilities(self) -> Sequence[OperatorCapability]:
        """Return the declared capabilities of a backend registry."""


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Backend-wide limits plus the operator candidate support matrix."""

    backend: str
    api_version: str
    supported_dtypes: frozenset[str]
    supported_layouts: frozenset[str]
    dynamic_shape_mode: str
    control_flow_mode: str
    operators: tuple[OperatorCapability, ...]

    @classmethod
    def from_registry(
        cls,
        registry: CapabilitySource,
        *,
        backend: str = "vulkan",
        api_version: str = "1.0",
        dynamic_shape_mode: str = "profile",
        control_flow_mode: str = "host-driven",
    ) -> "BackendCapabilities":
        operators = tuple(registry.capabilities())
        data_types = frozenset(
            data_type
            for capability in operators
            for data_type in capability.data_types
            if data_type != "*"
        )
        layouts = frozenset(
            layout
            for capability in operators
            for layout in capability.layouts
            if layout != "*"
        )
        return cls(
            backend=backend,
            api_version=api_version,
            supported_dtypes=data_types,
            supported_layouts=layouts,
            dynamic_shape_mode=dynamic_shape_mode,
            control_flow_mode=control_flow_mode,
            operators=operators,
        )

    def support_for(
        self,
        *,
        domain: str,
        op_type: str,
        opset_version: int,
        data_types: Sequence[str],
        layout: str = "contiguous",
    ) -> OperatorCapability | None:
        matches = [
            capability
            for capability in self.operators
            if capability.matches(
                domain=domain,
                op_type=op_type,
                opset_version=opset_version,
                data_types=data_types,
                layout=layout,
            )
        ]
        return max(matches, key=lambda item: item.min_opset, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "api_version": self.api_version,
            "supported_dtypes": sorted(self.supported_dtypes),
            "supported_layouts": sorted(self.supported_layouts),
            "dynamic_shape_mode": self.dynamic_shape_mode,
            "control_flow_mode": self.control_flow_mode,
            "operators": [item.to_dict() for item in self.operators],
        }
