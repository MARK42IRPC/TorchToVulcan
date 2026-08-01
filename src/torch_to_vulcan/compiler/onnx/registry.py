"""Version-aware registry for ONNX operator semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from onnx import TensorProto

from ..expr import (
    Assign,
    Binary,
    Call,
    ForEach,
    Literal,
    Name,
    OperatorProgram,
    Parameter,
    Property,
    TensorAccess,
    View,
    format_program,
)

Builder = Callable[[dict[str, Any]], OperatorProgram]


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    domain: str
    op_type: str
    since_version: int
    builder: Builder


@dataclass(frozen=True, slots=True)
class SemanticDefinition:
    key: str
    status: str
    category: str
    dialect: str
    pseudocode_en: str
    pseudocode_zh: str
    diagnostic_en: str = ""
    diagnostic_zh: str = ""
    source: str = "registry"
    confidence: str = "EXACT_RULE"


_REGISTRY: list[RegistryEntry] = []

_FLOAT8_TYPES = {
    "FLOAT8E4M3FN",
    "FLOAT8E4M3FNUZ",
    "FLOAT8E5M2",
    "FLOAT8E5M2FNUZ",
    "FLOAT8E8M0",
}


def register(domain: str, op_type: str, since_version: int) -> Callable[[Builder], Builder]:
    def decorator(builder: Builder) -> Builder:
        _REGISTRY.append(RegistryEntry(domain, op_type, since_version, builder))
        return builder

    return decorator


def lower_operator(
    domain: str,
    op_type: str,
    opset_version: int,
    attributes: dict[str, Any],
) -> SemanticDefinition:
    program = build_operator_program(domain, op_type, opset_version, attributes)
    if program is None:
        descriptor = {"domain": domain, "op_type": op_type, "opset": opset_version}
        key = semantic_key(descriptor)
        return SemanticDefinition(
            key=key,
            status="unsupported",
            category="UNDEFINED",
            dialect="TTV-Expr 0.1",
            pseudocode_en="",
            pseudocode_zh="",
            diagnostic_en="Operator semantics have not been defined yet.",
            diagnostic_zh="该算子的数学语义尚未定义。",
            source="unknown",
            confidence="UNKNOWN",
        )

    descriptor = {
        "domain": domain,
        "op_type": op_type,
        "opset": opset_version,
        "attributes": {
            name: value.value for name, value in program.attributes.items()
        },
    }
    return SemanticDefinition(
        key=semantic_key(descriptor),
        status="supported",
        category=program.category,
        dialect=program.dialect,
        pseudocode_en=format_program(program, "en"),
        pseudocode_zh=format_program(program, "zh"),
    )


def build_operator_program(
    domain: str,
    op_type: str,
    opset_version: int,
    attributes: dict[str, Any],
) -> OperatorProgram | None:
    candidates = [
        entry
        for entry in _REGISTRY
        if entry.domain == domain
        and entry.op_type == op_type
        and entry.since_version <= opset_version
    ]
    if not candidates:
        return None

    entry = max(candidates, key=lambda candidate: candidate.since_version)
    return entry.builder(attributes)


def semantic_key(descriptor: dict[str, Any]) -> str:
    encoded = json.dumps(descriptor, ensure_ascii=True, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    domain = descriptor["domain"] or "ai.onnx"
    return f"{domain}::{descriptor['op_type']}@{descriptor['opset']}:{digest}"


def _elementwise_program(name: str, operator: str) -> OperatorProgram:
    index = Name("index")
    output_shape = Property(Name("Y"), "shape")
    a_shape = Property(Name("A"), "shape")
    b_shape = Property(Name("B"), "shape")
    a_index = Name("a_index")
    b_index = Name("b_index")
    return OperatorProgram(
        name=name,
        category="ELEMENTWISE",
        inputs=(Parameter("A"), Parameter("B")),
        outputs=(Parameter("Y"),),
        constraints=(
            Binary("==", output_shape, Call("broadcast_shape", (a_shape, b_shape))),
        ),
        body=(
            ForEach(
                index="index",
                domain=Property(Name("Y"), "domain"),
                body=(
                    Assign(a_index, Call("broadcast_index", (index, a_shape, output_shape))),
                    Assign(b_index, Call("broadcast_index", (index, b_shape, output_shape))),
                    Assign(
                        TensorAccess("Y", index),
                        Binary(
                            operator,
                            TensorAccess("A", a_index),
                            TensorAccess("B", b_index),
                        ),
                    ),
                ),
            ),
        ),
    )


@register("", "Add", 7)
def _add(_: dict[str, Any]) -> OperatorProgram:
    return _elementwise_program("Add", "+")


@register("", "Mul", 7)
def _mul(_: dict[str, Any]) -> OperatorProgram:
    return _elementwise_program("Mul", "*")


@register("", "Relu", 1)
def _relu(_: dict[str, Any]) -> OperatorProgram:
    index = Name("index")
    return OperatorProgram(
        name="Relu",
        category="ELEMENTWISE",
        inputs=(Parameter("X"),),
        outputs=(Parameter("Y"),),
        constraints=(Binary("==", Property(Name("Y"), "shape"), Property(Name("X"), "shape")),),
        body=(
            ForEach(
                index="index",
                domain=Property(Name("Y"), "domain"),
                body=(
                    Assign(
                        TensorAccess("Y", index),
                        Call("maximum", (TensorAccess("X", index), Literal(0))),
                    ),
                ),
            ),
        ),
    )


def _cast_program(
    attributes: dict[str, Any],
    *,
    include_saturate: bool,
    include_round_mode: bool,
) -> OperatorProgram:
    raw_target = attributes.get("to", TensorProto.UNDEFINED)
    try:
        target = TensorProto.DataType.Name(int(raw_target))
    except (TypeError, ValueError):
        target = f"TYPE_{raw_target}"

    program_attributes = {"to": Literal(target)}
    cast_arguments: list[Name | TensorAccess] = [TensorAccess("X", Name("index")), Name("to")]
    if include_saturate and target in _FLOAT8_TYPES:
        program_attributes["saturate"] = Literal(bool(int(attributes.get("saturate", 1))))
        cast_arguments.append(Name("saturate"))
    if include_round_mode and target == "FLOAT8E8M0":
        round_mode = str(attributes.get("round_mode", "up"))
        program_attributes["round_mode"] = Literal(round_mode)
        cast_arguments.append(Name("round_mode"))

    index = Name("index")
    return OperatorProgram(
        name="Cast",
        category="CONVERSION",
        inputs=(Parameter("X", "Tensor<Source>"),),
        outputs=(Parameter("Y", f"Tensor<{target}>"),),
        attributes=program_attributes,
        constraints=(Binary("==", Property(Name("Y"), "shape"), Property(Name("X"), "shape")),),
        body=(
            ForEach(
                index="index",
                domain=Property(Name("Y"), "domain"),
                body=(
                    Assign(
                        TensorAccess("Y", index),
                        Call("cast", tuple(cast_arguments)),
                    ),
                ),
            ),
        ),
    )


@register("", "Cast", 1)
def _cast(attributes: dict[str, Any]) -> OperatorProgram:
    return _cast_program(attributes, include_saturate=False, include_round_mode=False)


@register("", "Cast", 19)
def _cast_float8(attributes: dict[str, Any]) -> OperatorProgram:
    return _cast_program(attributes, include_saturate=True, include_round_mode=False)


@register("", "Cast", 24)
def _cast_float8e8m0(attributes: dict[str, Any]) -> OperatorProgram:
    return _cast_program(attributes, include_saturate=True, include_round_mode=True)


@register("", "Reshape", 5)
def _reshape(attributes: dict[str, Any]) -> OperatorProgram:
    allowzero = int(attributes.get("allowzero", 0))
    return OperatorProgram(
        name="Reshape",
        category="VIEW",
        inputs=(Parameter("X"), Parameter("shape", "Tensor<int64>")),
        outputs=(Parameter("Y"),),
        attributes={"allowzero": Literal(allowzero)},
        constraints=(
            Binary(
                "==",
                Call("element_count", (Name("Y"),)),
                Call("element_count", (Name("X"),)),
            ),
        ),
        body=(View("Y", "X", Name("shape")),),
    )


@register("", "Transpose", 1)
def _transpose(attributes: dict[str, Any]) -> OperatorProgram:
    raw_perm = attributes.get("perm")
    perm = tuple(int(value) for value in raw_perm) if isinstance(raw_perm, list) else ()
    index = Name("output_index")
    attributes_ir = {"perm": Literal(perm if perm else "reverse_axes(X.rank)")}
    return OperatorProgram(
        name="Transpose",
        category="LAYOUT",
        inputs=(Parameter("X"),),
        outputs=(Parameter("Y"),),
        attributes=attributes_ir,
        body=(
            ForEach(
                index="output_index",
                domain=Property(Name("Y"), "domain"),
                body=(
                    Assign(
                        TensorAccess("Y", index),
                        TensorAccess(
                            "X",
                            Call("inverse_permute", (index, Name("perm"))),
                        ),
                    ),
                ),
            ),
        ),
    )


# Importing the declarative catalog executes its registrations after the core
# registry and shared expression helpers are fully initialized.
from . import basic as _basic  # noqa: E402,F401
