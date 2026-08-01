"""English and Chinese pretty-printer for TTV-Expr."""

from __future__ import annotations

from typing import Literal as TypingLiteral

from .ir import (
    Assign,
    Binary,
    Call,
    Expression,
    ForEach,
    Invoke,
    Literal,
    Name,
    OperatorProgram,
    Property,
    Statement,
    TensorAccess,
    View,
)

Language = TypingLiteral["en", "zh"]

_CALL_NAMES = {
    "en": {
        "broadcast_index": "broadcast_index",
        "broadcast_shape": "broadcast_shape",
        "batch_normalize": "batch_normalize",
        "cast_like": "cast_like",
        "cast": "cast",
        "clip": "clip",
        "concat": "concat",
        "constant": "constant",
        "convolution": "convolution",
        "convolution_transpose": "convolution_transpose",
        "cumulative_sum": "cumulative_sum",
        "dynamic_quantize_linear": "dynamic_quantize_linear",
        "element_count": "element_count",
        "fill": "fill",
        "gather": "gather",
        "gather_elements": "gather_elements",
        "instance_normalize": "instance_normalize",
        "inverse_permute": "inverse_permute",
        "layer_normalize": "layer_normalize",
        "matmul": "matmul",
        "matmul_integer": "matmul_integer",
        "maximum": "max",
        "pad": "pad",
        "range": "range",
        "reduce_l2": "reduce_l2",
        "reduce_mean": "reduce_mean",
        "reduce_sum": "reduce_sum",
        "shape_of": "shape_of",
        "argmax": "argmax",
        "resize": "resize",
        "scatter_elements": "scatter_elements",
        "slice": "slice",
        "softmax": "softmax",
        "split": "split",
        "tile": "tile",
        "top_k": "top_k",
        "where": "where",
    },
    "zh": {
        "abs": "绝对值",
        "batch_normalize": "批归一化",
        "broadcast_to": "广播到",
        "broadcast_index": "广播索引",
        "broadcast_shape": "广播形状",
        "cast": "类型转换",
        "cast_like": "按目标类型转换",
        "clip": "范围截断",
        "concat": "拼接",
        "constant": "常量",
        "convolution": "卷积",
        "convolution_transpose": "转置卷积",
        "cumulative_sum": "累积求和",
        "dynamic_quantize_linear": "动态线性量化",
        "element_count": "元素数量",
        "fill": "填充",
        "erf": "误差函数",
        "exp": "指数",
        "floor": "向下取整",
        "gather": "索引收集",
        "gather_elements": "按元素收集",
        "instance_normalize": "实例归一化",
        "inverse_permute": "逆置换",
        "leaky_relu": "带泄漏整流",
        "layer_normalize": "层归一化",
        "log": "对数",
        "logical_not": "逻辑非",
        "matmul": "矩阵乘法",
        "matmul_integer": "整数矩阵乘法",
        "maximum": "最大值",
        "negate": "取负",
        "pad": "边界填充",
        "range": "数值范围",
        "reduce_l2": "二范数归约",
        "reduce_mean": "均值归约",
        "reduce_sum": "求和归约",
        "shape_of": "获取形状",
        "argmax": "最大值索引",
        "resize": "尺寸变换",
        "scatter_elements": "按元素散布",
        "sigmoid": "逻辑函数",
        "sin": "正弦",
        "slice": "切片",
        "softmax": "归一化指数",
        "softplus": "柔性整流",
        "split": "拆分",
        "sqrt": "平方根",
        "squeeze_view": "移除单维视图",
        "tanh": "双曲正切",
        "tile": "平铺",
        "top_k": "前K项",
        "unsqueeze_view": "增加维度视图",
        "where": "条件选择",
        "cos": "余弦",
    },
}


def format_program(program: OperatorProgram, language: Language) -> str:
    labels = {
        "en": {
            "operator": "operator",
            "input": "input",
            "output": "output",
            "attribute": "attribute",
            "constraint": "constraint",
            "known_input": "known input",
            "parallel": "parallel",
            "in": "in",
            "view": "view",
        },
        "zh": {
            "operator": "算子",
            "input": "输入",
            "output": "输出",
            "attribute": "属性",
            "constraint": "约束",
            "known_input": "已知输入",
            "parallel": "并行遍历",
            "in": "属于",
            "view": "视图",
        },
    }[language]
    lines = [f"{labels['operator']} {program.name}"]
    lines.extend(
        f"{labels['input']}  {parameter.name}: {parameter.tensor_type}"
        for parameter in program.inputs
    )
    lines.extend(
        f"{labels['output']} {parameter.name}: {parameter.tensor_type}"
        for parameter in program.outputs
    )
    if program.attributes:
        lines.append("")
        lines.extend(
            f"{labels['attribute']} {name} = {_format_expression(value, language)}"
            for name, value in program.attributes.items()
        )
    if program.bindings:
        lines.append("")
        lines.extend(
            f"{labels['known_input']} {name} = {_format_expression(value, language)}"
            for name, value in program.bindings.items()
        )
    if program.constraints:
        lines.append("")
        lines.extend(
            f"{labels['constraint']} {_format_expression(value, language)}"
            for value in program.constraints
        )
    lines.append("")
    for statement in program.body:
        lines.extend(_format_statement(statement, language, labels, 0))
    return "\n".join(lines)


def _format_statement(
    statement: Statement,
    language: Language,
    labels: dict[str, str],
    indentation: int,
) -> list[str]:
    prefix = "    " * indentation
    if isinstance(statement, Assign):
        return [
            f"{prefix}{_format_expression(statement.target, language)} = "
            f"{_format_expression(statement.value, language)}"
        ]
    if isinstance(statement, View):
        shape = _format_expression(statement.shape, language)
        return [
            f"{prefix}{labels['view']} {statement.output} = "
            f"reshape_view({statement.source}, {shape})"
        ]
    if isinstance(statement, ForEach):
        domain = _format_expression(statement.domain, language)
        if language == "zh":
            header = f"{prefix}{labels['parallel']} {domain} 中的 {statement.index}:"
        else:
            header = (
                f"{prefix}{labels['parallel']} {statement.index} "
                f"{labels['in']} {domain}:"
            )
        lines = [header]
        for child in statement.body:
            lines.extend(_format_statement(child, language, labels, indentation + 1))
        return lines
    if isinstance(statement, Invoke):
        outputs = ", ".join(statement.outputs)
        inputs = ", ".join(statement.inputs)
        attributes = ", ".join(
            f"{name}={_format_expression(value, language)}"
            for name, value in statement.attributes
        )
        arguments = ", ".join(value for value in (inputs, attributes) if value)
        return [f"{prefix}{outputs} = {statement.operator}({arguments})"]
    raise TypeError(f"unsupported statement: {type(statement).__name__}")


def _format_expression(expression: Expression, language: Language) -> str:
    if isinstance(expression, Name):
        return expression.value
    if isinstance(expression, Literal):
        if isinstance(expression.value, str):
            return expression.value
        if isinstance(expression.value, tuple):
            return "[" + ", ".join(str(value) for value in expression.value) + "]"
        if isinstance(expression.value, bool):
            return str(expression.value).lower()
        return repr(expression.value)
    if isinstance(expression, Property):
        return f"{_format_expression(expression.value, language)}.{expression.name}"
    if isinstance(expression, TensorAccess):
        return f"{expression.tensor}[{_format_expression(expression.index, language)}]"
    if isinstance(expression, Call):
        name = _CALL_NAMES[language].get(expression.function, expression.function)
        arguments = ", ".join(
            _format_expression(argument, language) for argument in expression.arguments
        )
        return f"{name}({arguments})"
    if isinstance(expression, Binary):
        left = _format_expression(expression.left, language)
        right = _format_expression(expression.right, language)
        return f"{left} {expression.operator} {right}"
    raise TypeError(f"unsupported expression: {type(expression).__name__}")
