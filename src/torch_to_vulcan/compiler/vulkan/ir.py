"""Hardware-facing schedule IR for Vulkan compute dispatches."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BufferBinding:
    binding: int
    name: str
    access: str
    data_type: str


@dataclass(frozen=True, slots=True)
class ShaderModule:
    name: str
    source: str
    entry_point: str = "main"
    language: str = "GLSL 450"


@dataclass(frozen=True, slots=True)
class DispatchStep:
    shader: ShaderModule
    workgroups: tuple[int, int, int]
    bindings: tuple[BufferBinding, ...]
    push_constants: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    kernel_id: str
    operator: str
    steps: tuple[DispatchStep, ...]
    metadata_only: bool = False
    notes: tuple[str, ...] = ()

    @property
    def dispatch_count(self) -> int:
        return len(self.steps)
