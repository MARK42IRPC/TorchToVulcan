"""Structured tensor-expression language used before hardware scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Name:
    value: str


@dataclass(frozen=True, slots=True)
class Literal:
    value: bool | int | float | str | tuple[bool | int | float | str, ...]


@dataclass(frozen=True, slots=True)
class Property:
    value: "Expression"
    name: str


@dataclass(frozen=True, slots=True)
class TensorAccess:
    tensor: str
    index: "Expression"


@dataclass(frozen=True, slots=True)
class Call:
    function: str
    arguments: tuple["Expression", ...]


@dataclass(frozen=True, slots=True)
class Binary:
    operator: str
    left: "Expression"
    right: "Expression"


Expression: TypeAlias = Name | Literal | Property | TensorAccess | Call | Binary


@dataclass(frozen=True, slots=True)
class Assign:
    target: Name | TensorAccess
    value: Expression


@dataclass(frozen=True, slots=True)
class ForEach:
    index: str
    domain: Expression
    body: tuple["Statement", ...]


@dataclass(frozen=True, slots=True)
class View:
    output: str
    source: str
    shape: Expression


@dataclass(frozen=True, slots=True)
class Invoke:
    outputs: tuple[str, ...]
    operator: str
    inputs: tuple[str, ...]
    attributes: tuple[tuple[str, Literal], ...] = ()


Statement: TypeAlias = Assign | ForEach | View | Invoke


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    tensor_type: str = "Tensor<T>"


@dataclass(frozen=True, slots=True)
class OperatorProgram:
    name: str
    category: str
    inputs: tuple[Parameter, ...]
    outputs: tuple[Parameter, ...]
    body: tuple[Statement, ...]
    constraints: tuple[Expression, ...] = ()
    attributes: dict[str, Literal] = field(default_factory=dict)
    bindings: dict[str, Literal] = field(default_factory=dict)
    dialect: str = "TTV-Expr 0.1"
