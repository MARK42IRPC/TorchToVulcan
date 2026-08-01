"""Context-aware ONNX semantic resolution and FunctionProto decomposition."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable

from onnx import AttributeProto, FunctionProto, TensorProto, defs, helper, numpy_helper

from ..expr import Invoke, Literal, OperatorProgram, Parameter, format_program
from .registry import (
    SemanticDefinition,
    build_operator_program,
    semantic_key,
)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    data_type: str
    shape: tuple[str, ...]

    @property
    def display_type(self) -> str:
        if self.data_type == "UNKNOWN" and not self.shape:
            return "Tensor<UNKNOWN>[?]"
        dimensions = ", ".join(self.shape)
        return f"Tensor<{self.data_type}>[{dimensions}]"

    @property
    def semantic_type(self) -> str:
        return f"Tensor<{self.data_type}>"


@dataclass(frozen=True, slots=True)
class NodeContext:
    domain: str
    op_type: str
    opset_version: int
    attributes: dict[str, Any]
    inputs: tuple[TensorSpec, ...] = ()
    outputs: tuple[TensorSpec, ...] = ()
    constant_inputs: dict[str, Any] | None = None
    overload: str = ""


class SemanticResolver:
    def __init__(self, functions: Iterable[FunctionProto] = ()) -> None:
        self._functions = {
            (function.domain, function.name, getattr(function, "overload", "")): function
            for function in functions
        }

    def resolve(self, context: NodeContext) -> SemanticDefinition:
        local_function = self._functions.get(
            (context.domain, context.op_type, context.overload)
        )
        if local_function is not None:
            return _function_definition(context, local_function, "model_function")

        program = build_operator_program(
            context.domain,
            context.op_type,
            context.opset_version,
            context.attributes,
        )
        if program is not None:
            return _program_definition(context, _specialize_program(program, context))

        schema_function = _schema_function(context)
        if schema_function is not None:
            function, defaults = schema_function
            resolved_context = replace(
                context,
                attributes={**defaults, **context.attributes},
            )
            return _function_definition(
                resolved_context,
                function,
                "schema_function",
            )

        descriptor = _context_descriptor(context, "unknown")
        return SemanticDefinition(
            key=semantic_key(descriptor),
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


def _program_definition(
    context: NodeContext,
    program: OperatorProgram,
) -> SemanticDefinition:
    descriptor = _context_descriptor(context, "registry")
    descriptor["semantic_attributes"] = {
        name: value.value for name, value in program.attributes.items()
    }
    descriptor["bindings"] = {
        name: value.value for name, value in program.bindings.items()
    }
    return SemanticDefinition(
        key=semantic_key(descriptor),
        status="supported",
        category=program.category,
        dialect=program.dialect,
        pseudocode_en=format_program(program, "en"),
        pseudocode_zh=format_program(program, "zh"),
        source="registry",
        confidence="EXACT_RULE",
    )


def _function_definition(
    context: NodeContext,
    function: FunctionProto,
    source: str,
) -> SemanticDefinition:
    program = _function_program(context, function)
    descriptor = _context_descriptor(context, source)
    descriptor["function"] = hashlib.sha256(function.SerializeToString()).hexdigest()[:16]
    return SemanticDefinition(
        key=semantic_key(descriptor),
        status="supported",
        category="COMPOSITE",
        dialect=program.dialect,
        pseudocode_en=format_program(program, "en"),
        pseudocode_zh=format_program(program, "zh"),
        source=source,
        confidence="EXACT_FUNCTION",
    )


def _specialize_program(program: OperatorProgram, context: NodeContext) -> OperatorProgram:
    inputs = list(program.inputs)
    outputs = list(program.outputs)
    if len(inputs) == len(context.inputs):
        inputs = [
            replace(parameter, tensor_type=spec.semantic_type)
            for parameter, spec in zip(inputs, context.inputs, strict=True)
        ]
    if len(outputs) == len(context.outputs):
        outputs = [
            replace(parameter, tensor_type=spec.semantic_type)
            for parameter, spec in zip(outputs, context.outputs, strict=True)
        ]

    bindings = dict(program.bindings)
    constants = context.constant_inputs or {}
    for index, parameter in enumerate(program.inputs):
        if index >= len(context.inputs):
            break
        actual_name = context.inputs[index].name
        if actual_name in constants:
            bindings[parameter.name] = _literal(constants[actual_name])

    return replace(
        program,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        bindings=bindings,
    )


def _function_program(context: NodeContext, function: FunctionProto) -> OperatorProgram:
    inputs = tuple(
        Parameter(name, _spec_type(context.inputs, index))
        for index, name in enumerate(function.input)
    )
    outputs = tuple(
        Parameter(name, _spec_type(context.outputs, index))
        for index, name in enumerate(function.output)
    )
    defaults = _function_defaults(function)
    attributes = {
        name: _literal(value)
        for name, value in {**defaults, **context.attributes}.items()
    }
    body = tuple(
        Invoke(
            outputs=tuple(node.output),
            operator=f"{node.domain}::{node.op_type}" if node.domain else node.op_type,
            inputs=tuple(value for value in node.input if value),
            attributes=tuple(
                (attribute.name, _function_attribute(attribute, context.attributes, defaults))
                for attribute in node.attribute
            ),
        )
        for node in function.node
    )
    bindings: dict[str, Literal] = {}
    constants = context.constant_inputs or {}
    for index, formal_name in enumerate(function.input):
        if index < len(context.inputs) and context.inputs[index].name in constants:
            bindings[formal_name] = _literal(constants[context.inputs[index].name])
    return OperatorProgram(
        name=context.op_type,
        category="COMPOSITE",
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        bindings=bindings,
        body=body,
    )


def _schema_function(
    context: NodeContext,
) -> tuple[FunctionProto, dict[str, Any]] | None:
    try:
        schema = defs.get_schema(
            context.op_type,
            context.opset_version,
            context.domain,
        )
    except defs.SchemaError:
        return None
    if not schema.has_function:
        return None
    defaults = {
        name: _attribute_value(attribute.default_value)
        for name, attribute in schema.attributes.items()
        if attribute.default_value.name
    }
    return schema.function_body, defaults


def _function_defaults(function: FunctionProto) -> dict[str, Any]:
    return {
        attribute.name: _attribute_value(attribute)
        for attribute in function.attribute_proto
    }


def _function_attribute(
    attribute: AttributeProto,
    attributes: dict[str, Any],
    defaults: dict[str, Any],
) -> Literal:
    if attribute.ref_attr_name:
        value = attributes.get(attribute.ref_attr_name, defaults.get(attribute.ref_attr_name))
        return _literal(value)
    return _literal(_attribute_value(attribute))


def _attribute_value(attribute: AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, TensorProto):
        return _tensor_value(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [
            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
            for item in value
        ]
    return value


def _tensor_value(tensor: TensorProto) -> Any:
    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        try:
            data_type = TensorProto.DataType.Name(tensor.data_type)
        except ValueError:
            data_type = f"TYPE_{tensor.data_type}"
        return {"data_type": data_type, "shape": list(tensor.dims)}
    element_count = 1
    for dimension in tensor.dims:
        element_count *= dimension
    if element_count <= 16:
        try:
            array = numpy_helper.to_array(tensor).reshape(-1).tolist()
            if len(array) == 1 and not tensor.dims:
                return array[0]
            return array
        except (TypeError, ValueError):
            pass
    try:
        data_type = TensorProto.DataType.Name(tensor.data_type)
    except ValueError:
        data_type = f"TYPE_{tensor.data_type}"
    return {"data_type": data_type, "shape": list(tensor.dims)}


def _literal(value: Any) -> Literal:
    if isinstance(value, (list, tuple)):
        return Literal(tuple(_scalar(item) for item in value))
    if isinstance(value, dict) and "data_type" in value:
        shape = ", ".join(str(item) for item in value.get("shape", ()))
        return Literal(f"Tensor<{value['data_type']}>[{shape}]")
    return Literal(_scalar(value))


def _scalar(value: Any) -> bool | int | float | str:
    if isinstance(value, (bool, int, float, str)):
        return value
    return "none" if value is None else str(value)


def _spec_type(specs: tuple[TensorSpec, ...], index: int) -> str:
    return specs[index].semantic_type if index < len(specs) else "Tensor<UNKNOWN>"


def _context_descriptor(context: NodeContext, source: str) -> dict[str, Any]:
    return {
        "domain": context.domain,
        "op_type": context.op_type,
        "opset": context.opset_version,
        "source": source,
        "overload": context.overload,
        "attributes": context.attributes,
        "inputs": [spec.semantic_type for spec in context.inputs],
        "outputs": [spec.semantic_type for spec in context.outputs],
        "constant_inputs": context.constant_inputs or {},
    }
