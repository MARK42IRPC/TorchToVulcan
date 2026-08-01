"""Declarative semantics for foundational ONNX operators."""

from __future__ import annotations

from typing import Any

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
)
from .registry import register


def _literal(value: Any) -> Literal:
    if isinstance(value, (list, tuple)):
        return Literal(tuple(value))
    if isinstance(value, dict) and "data_type" in value:
        shape = ", ".join(str(dimension) for dimension in value.get("shape", ()))
        return Literal(f"Tensor<{value['data_type']}>[{shape}]")
    if isinstance(value, (bool, int, float, str)):
        return Literal(value)
    return Literal("none" if value is None else str(value))


def _attributes(
    source: dict[str, Any],
    defaults: tuple[tuple[str, Any], ...],
) -> dict[str, Literal]:
    return {name: _literal(source.get(name, default)) for name, default in defaults}


def _same_shape() -> Binary:
    return Binary("==", Property(Name("Y"), "shape"), Property(Name("X"), "shape"))


def _unary_program(
    name: str,
    function: str,
    *,
    output_type: str = "Tensor<T>",
) -> OperatorProgram:
    index = Name("index")
    return OperatorProgram(
        name=name,
        category="ELEMENTWISE",
        inputs=(Parameter("X"),),
        outputs=(Parameter("Y", output_type),),
        constraints=(_same_shape(),),
        body=(
            ForEach(
                index="index",
                domain=Property(Name("Y"), "domain"),
                body=(
                    Assign(
                        TensorAccess("Y", index),
                        Call(function, (TensorAccess("X", index),)),
                    ),
                ),
            ),
        ),
    )


def _binary_program(
    name: str,
    operator: str,
    *,
    output_type: str = "Tensor<T>",
) -> OperatorProgram:
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
        outputs=(Parameter("Y", output_type),),
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
                        Binary(operator, TensorAccess("A", a_index), TensorAccess("B", b_index)),
                    ),
                ),
            ),
        ),
    )


def _call_program(
    name: str,
    category: str,
    function: str,
    inputs: tuple[Parameter, ...],
    arguments: tuple[str, ...],
    *,
    attributes: dict[str, Literal] | None = None,
    output: Parameter = Parameter("Y"),
    target: str = "Y",
) -> OperatorProgram:
    return OperatorProgram(
        name=name,
        category=category,
        inputs=inputs,
        outputs=(output,),
        attributes=attributes or {},
        body=(Assign(Name(target), Call(function, tuple(Name(value) for value in arguments))),),
    )


def _register_unary(name: str, function: str, *, output_type: str = "Tensor<T>") -> None:
    def builder(_: dict[str, Any]) -> OperatorProgram:
        return _unary_program(name, function, output_type=output_type)

    register("", name, 1)(builder)


def _register_binary(name: str, operator: str, *, output_type: str = "Tensor<T>") -> None:
    def builder(_: dict[str, Any]) -> OperatorProgram:
        return _binary_program(name, operator, output_type=output_type)

    register("", name, 7)(builder)


for _name, _function in (
    ("Abs", "abs"),
    ("Erf", "erf"),
    ("Exp", "exp"),
    ("Floor", "floor"),
    ("Log", "log"),
    ("Neg", "negate"),
    ("Sigmoid", "sigmoid"),
    ("Sin", "sin"),
    ("Cos", "cos"),
    ("Softplus", "softplus"),
    ("Sqrt", "sqrt"),
    ("Tanh", "tanh"),
):
    _register_unary(_name, _function)

_register_unary("Not", "logical_not", output_type="Tensor<bool>")

for _name, _operator in (
    ("Sub", "-"),
    ("Div", "/"),
    ("Pow", "**"),
):
    _register_binary(_name, _operator)

for _name, _operator in (
    ("Equal", "=="),
    ("Greater", ">"),
    ("Less", "<"),
    ("GreaterOrEqual", ">="),
    ("LessOrEqual", "<="),
    ("And", "and"),
    ("Or", "or"),
    ("Xor", "xor"),
):
    _register_binary(_name, _operator, output_type="Tensor<bool>")


@register("", "Identity", 1)
def _identity(_: dict[str, Any]) -> OperatorProgram:
    return OperatorProgram(
        name="Identity",
        category="VIEW",
        inputs=(Parameter("X"),),
        outputs=(Parameter("Y"),),
        body=(Assign(Name("Y"), Name("X")),),
    )


@register("", "LeakyRelu", 1)
def _leaky_relu(attributes: dict[str, Any]) -> OperatorProgram:
    program = _unary_program("LeakyRelu", "leaky_relu")
    return OperatorProgram(
        name=program.name,
        category=program.category,
        inputs=program.inputs,
        outputs=program.outputs,
        attributes=_attributes(attributes, (("alpha", 0.01),)),
        constraints=program.constraints,
        body=program.body,
    )


@register("", "Where", 9)
def _where(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Where",
        "ELEMENTWISE",
        "where",
        (Parameter("condition", "Tensor<bool>"), Parameter("A"), Parameter("B")),
        ("condition", "A", "B"),
    )


@register("", "Max", 6)
def _max(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Max",
        "ELEMENTWISE",
        "maximum",
        (Parameter("inputs", "Sequence<Tensor<T>>"),),
        ("inputs",),
    )


@register("", "Clip", 11)
def _clip(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Clip",
        "ELEMENTWISE",
        "clip",
        (Parameter("X"), Parameter("min", "Scalar<T>?"), Parameter("max", "Scalar<T>?")),
        ("X", "min", "max"),
    )


@register("", "Constant", 1)
def _constant(attributes: dict[str, Any]) -> OperatorProgram:
    value_name = next((name for name in attributes if name.startswith("value")), "value")
    value = attributes.get(value_name, "value")
    return _call_program(
        "Constant",
        "CONSTANT",
        "constant",
        (),
        ("value",),
        attributes={"value": _literal(value)},
        output=Parameter("Y", "Tensor<Constant>"),
    )


@register("", "ConstantOfShape", 9)
def _constant_of_shape(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ConstantOfShape",
        "CONSTANT",
        "fill",
        (Parameter("shape", "Tensor<int64>"),),
        ("shape", "value"),
        attributes={"value": _literal(attributes.get("value", "0<float32>"))},
    )


@register("", "Shape", 1)
def _shape(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Shape",
        "SHAPE",
        "shape_of",
        (Parameter("X"),),
        ("X",),
        output=Parameter("Y", "Tensor<int64>"),
    )


@register("", "Shape", 15)
def _shape_slice(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Shape",
        "SHAPE",
        "shape_of",
        (Parameter("X"),),
        ("X", "start", "end"),
        attributes=_attributes(attributes, (("start", 0), ("end", "rank(X)"))),
        output=Parameter("Y", "Tensor<int64>"),
    )


@register("", "Unsqueeze", 13)
def _unsqueeze(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Unsqueeze",
        "VIEW",
        "unsqueeze_view",
        (Parameter("X"), Parameter("axes", "Tensor<int64>")),
        ("X", "axes"),
    )


@register("", "Squeeze", 13)
def _squeeze(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Squeeze",
        "VIEW",
        "squeeze_view",
        (Parameter("X"), Parameter("axes", "Tensor<int64>?")),
        ("X", "axes"),
    )


@register("", "Expand", 8)
def _expand(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Expand",
        "VIEW",
        "broadcast_to",
        (Parameter("X"), Parameter("shape", "Tensor<int64>")),
        ("X", "shape"),
    )


@register("", "Concat", 4)
def _concat(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Concat",
        "LAYOUT",
        "concat",
        (Parameter("inputs", "Sequence<Tensor<T>>"),),
        ("inputs", "axis"),
        attributes=_attributes(attributes, (("axis", "required"),)),
    )


@register("", "Slice", 10)
def _slice(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Slice",
        "LAYOUT",
        "slice",
        (
            Parameter("X"),
            Parameter("starts", "Tensor<int64>"),
            Parameter("ends", "Tensor<int64>"),
            Parameter("axes", "Tensor<int64>?"),
            Parameter("steps", "Tensor<int64>?"),
        ),
        ("X", "starts", "ends", "axes", "steps"),
    )


@register("", "Gather", 1)
def _gather(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Gather",
        "INDEXING",
        "gather",
        (Parameter("data"), Parameter("indices", "Tensor<int>")),
        ("data", "indices", "axis"),
        attributes=_attributes(attributes, (("axis", 0),)),
    )


@register("", "GatherElements", 11)
def _gather_elements(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "GatherElements",
        "INDEXING",
        "gather_elements",
        (Parameter("data"), Parameter("indices", "Tensor<int>")),
        ("data", "indices", "axis"),
        attributes=_attributes(attributes, (("axis", 0),)),
    )


@register("", "Split", 13)
def _split(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Split",
        "LAYOUT",
        "split",
        (Parameter("X"), Parameter("split", "Tensor<int64>?")),
        ("X", "split", "axis", "num_outputs"),
        attributes=_attributes(attributes, (("axis", 0), ("num_outputs", "inferred"))),
        output=Parameter("outputs", "Sequence<Tensor<T>>"),
        target="outputs",
    )


@register("", "Pad", 11)
def _pad(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Pad",
        "LAYOUT",
        "pad",
        (
            Parameter("X"),
            Parameter("pads", "Tensor<int64>"),
            Parameter("constant_value", "Scalar<T>?"),
            Parameter("axes", "Tensor<int64>?"),
        ),
        ("X", "pads", "constant_value", "axes", "mode"),
        attributes=_attributes(attributes, (("mode", "constant"),)),
    )


@register("", "Tile", 6)
def _tile(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Tile",
        "LAYOUT",
        "tile",
        (Parameter("X"), Parameter("repeats", "Tensor<int64>")),
        ("X", "repeats"),
    )


@register("", "Range", 11)
def _range(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Range",
        "GENERATOR",
        "range",
        (
            Parameter("start", "Scalar<T>"),
            Parameter("limit", "Scalar<T>"),
            Parameter("delta", "Scalar<T>"),
        ),
        ("start", "limit", "delta"),
    )


@register("", "ReduceSum", 13)
def _reduce_sum(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ReduceSum",
        "REDUCTION",
        "reduce_sum",
        (Parameter("X"), Parameter("axes", "Tensor<int64>?")),
        ("X", "axes", "keepdims", "noop_with_empty_axes"),
        attributes=_attributes(attributes, (("keepdims", 1), ("noop_with_empty_axes", 0))),
    )


@register("", "ReduceMean", 1)
def _reduce_mean(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ReduceMean",
        "REDUCTION",
        "reduce_mean",
        (Parameter("X"),),
        ("X", "axes", "keepdims"),
        attributes=_attributes(attributes, (("axes", "all_axes"), ("keepdims", 1))),
    )


@register("", "Softmax", 13)
def _softmax(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Softmax",
        "REDUCTION",
        "softmax",
        (Parameter("X"),),
        ("X", "axis"),
        attributes=_attributes(attributes, (("axis", -1),)),
    )


@register("", "MatMul", 1)
def _matmul(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "MatMul",
        "LINEAR_ALGEBRA",
        "matmul",
        (Parameter("A"), Parameter("B")),
        ("A", "B"),
    )


@register("", "MatMulInteger", 10)
def _matmul_integer(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "MatMulInteger",
        "LINEAR_ALGEBRA",
        "matmul_integer",
        (
            Parameter("A", "Tensor<int8|uint8>"),
            Parameter("B", "Tensor<int8|uint8>"),
            Parameter("a_zero_point", "Scalar<int>?"),
            Parameter("b_zero_point", "Scalar<int>?"),
        ),
        ("A", "B", "a_zero_point", "b_zero_point"),
        output=Parameter("Y", "Tensor<int32>"),
    )


@register("", "Gemm", 1)
def _gemm(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Gemm",
        "LINEAR_ALGEBRA",
        "gemm",
        (Parameter("A"), Parameter("B"), Parameter("C", "Tensor<T>?")),
        ("A", "B", "C", "alpha", "beta", "transA", "transB"),
        attributes=_attributes(
            attributes,
            (("alpha", 1.0), ("beta", 1.0), ("transA", 0), ("transB", 0)),
        ),
    )


@register("", "Conv", 1)
def _conv(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Conv",
        "CONVOLUTION",
        "convolution",
        (Parameter("X"), Parameter("W"), Parameter("B", "Tensor<T>?")),
        ("X", "W", "B", "strides", "pads", "dilations", "group"),
        attributes=_attributes(
            attributes,
            (
                ("strides", "ones(spatial_rank)"),
                ("pads", "zeros(2 * spatial_rank)"),
                ("dilations", "ones(spatial_rank)"),
                ("group", 1),
            ),
        ),
    )


@register("", "BatchNormalization", 9)
def _batch_normalization(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "BatchNormalization",
        "NORMALIZATION",
        "batch_normalize",
        (
            Parameter("X"),
            Parameter("scale"),
            Parameter("bias"),
            Parameter("input_mean"),
            Parameter("input_variance"),
        ),
        ("X", "scale", "bias", "input_mean", "input_variance", "epsilon"),
        attributes=_attributes(attributes, (("epsilon", 1e-5),)),
    )


@register("", "LayerNormalization", 17)
def _layer_normalization(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "LayerNormalization",
        "NORMALIZATION",
        "layer_normalize",
        (Parameter("X"), Parameter("scale"), Parameter("B", "Tensor<T>?")),
        ("X", "scale", "B", "axis", "epsilon"),
        attributes=_attributes(attributes, (("axis", -1), ("epsilon", 1e-5))),
    )


@register("", "DynamicQuantizeLinear", 11)
def _dynamic_quantize_linear(_: dict[str, Any]) -> OperatorProgram:
    return OperatorProgram(
        name="DynamicQuantizeLinear",
        category="QUANTIZATION",
        inputs=(Parameter("X", "Tensor<float>"),),
        outputs=(
            Parameter("Y", "Tensor<uint8>"),
            Parameter("Y_scale", "Scalar<float>"),
            Parameter("Y_zero_point", "Scalar<uint8>"),
        ),
        body=(
            Assign(
                Name("(Y, Y_scale, Y_zero_point)"),
                Call("dynamic_quantize_linear", (Name("X"),)),
            ),
        ),
    )


@register("", "CumSum", 11)
def _cum_sum(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "CumSum",
        "SCAN",
        "cumulative_sum",
        (Parameter("X"), Parameter("axis", "Scalar<int>")),
        ("X", "axis", "exclusive", "reverse"),
        attributes=_attributes(attributes, (("exclusive", 0), ("reverse", 0))),
    )


@register("", "ArgMax", 13)
def _argmax(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ArgMax",
        "REDUCTION",
        "argmax",
        (Parameter("X"),),
        ("X", "axis", "keepdims", "select_last_index"),
        attributes=_attributes(
            attributes,
            (("axis", 0), ("keepdims", 1), ("select_last_index", 0)),
        ),
        output=Parameter("Y", "Tensor<int64>"),
    )


@register("", "TopK", 11)
def _top_k(attributes: dict[str, Any]) -> OperatorProgram:
    return OperatorProgram(
        name="TopK",
        category="SELECTION",
        inputs=(Parameter("X"), Parameter("K", "Scalar<int64>")),
        outputs=(Parameter("Values"), Parameter("Indices", "Tensor<int64>")),
        attributes=_attributes(
            attributes,
            (("axis", -1), ("largest", 1), ("sorted", 1)),
        ),
        body=(
            Assign(
                Name("(Values, Indices)"),
                Call(
                    "top_k",
                    (Name("X"), Name("K"), Name("axis"), Name("largest"), Name("sorted")),
                ),
            ),
        ),
    )


@register("", "ScatterElements", 16)
def _scatter_elements(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ScatterElements",
        "INDEXING",
        "scatter_elements",
        (Parameter("data"), Parameter("indices", "Tensor<int>"), Parameter("updates")),
        ("data", "indices", "updates", "axis", "reduction"),
        attributes=_attributes(attributes, (("axis", 0), ("reduction", "none"))),
    )


@register("", "Resize", 13)
def _resize(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "Resize",
        "RESAMPLING",
        "resize",
        (
            Parameter("X"),
            Parameter("roi", "Tensor<float>?"),
            Parameter("scales", "Tensor<float>?"),
            Parameter("sizes", "Tensor<int64>?"),
        ),
        ("X", "roi", "scales", "sizes", "mode", "coordinate_transformation_mode"),
        attributes=_attributes(
            attributes,
            (("mode", "nearest"), ("coordinate_transformation_mode", "half_pixel")),
        ),
    )


@register("", "PRelu", 7)
def _prelu(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "PRelu",
        "ELEMENTWISE",
        "prelu",
        (Parameter("X"), Parameter("slope")),
        ("X", "slope"),
    )


@register("", "InstanceNormalization", 6)
def _instance_normalization(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "InstanceNormalization",
        "NORMALIZATION",
        "instance_normalize",
        (Parameter("X"), Parameter("scale"), Parameter("B")),
        ("X", "scale", "B", "epsilon"),
        attributes=_attributes(attributes, (("epsilon", 1e-5),)),
    )


@register("", "CastLike", 15)
def _cast_like(_: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "CastLike",
        "CONVERSION",
        "cast_like",
        (Parameter("X"), Parameter("target_type")),
        ("X", "target_type"),
    )


@register("", "CastLike", 19)
def _cast_like_float8(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "CastLike",
        "CONVERSION",
        "cast_like",
        (Parameter("X"), Parameter("target_type")),
        ("X", "target_type", "saturate"),
        attributes=_attributes(attributes, (("saturate", 1),)),
    )


@register("", "ReduceL2", 1)
def _reduce_l2(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ReduceL2",
        "REDUCTION",
        "reduce_l2",
        (Parameter("X"),),
        ("X", "axes", "keepdims"),
        attributes=_attributes(attributes, (("axes", "all_axes"), ("keepdims", 1))),
    )


@register("", "ReduceL2", 18)
def _reduce_l2_axes_input(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ReduceL2",
        "REDUCTION",
        "reduce_l2",
        (Parameter("X"), Parameter("axes", "Tensor<int64>?")),
        ("X", "axes", "keepdims", "noop_with_empty_axes"),
        attributes=_attributes(attributes, (("keepdims", 1), ("noop_with_empty_axes", 0))),
    )


@register("", "ConvTranspose", 1)
def _conv_transpose(attributes: dict[str, Any]) -> OperatorProgram:
    return _call_program(
        "ConvTranspose",
        "CONVOLUTION",
        "convolution_transpose",
        (Parameter("X"), Parameter("W"), Parameter("B", "Tensor<T>?")),
        ("X", "W", "B", "strides", "pads", "dilations", "group", "output_padding"),
        attributes=_attributes(
            attributes,
            (
                ("strides", "ones(spatial_rank)"),
                ("pads", "zeros(2 * spatial_rank)"),
                ("dilations", "ones(spatial_rank)"),
                ("group", 1),
                ("output_padding", "zeros(spatial_rank)"),
            ),
        ),
    )
