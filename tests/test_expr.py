from __future__ import annotations

import unittest

from onnx import TensorProto

from torch_to_vulcan.compiler.onnx import (
    NodeContext,
    SemanticResolver,
    TensorSpec,
    lower_operator,
)


class TtvExprTests(unittest.TestCase):
    def test_formats_relu_in_english_and_chinese(self) -> None:
        definition = lower_operator("", "Relu", 18, {})

        self.assertEqual(definition.status, "supported")
        self.assertEqual(definition.category, "ELEMENTWISE")
        self.assertIn("operator Relu", definition.pseudocode_en)
        self.assertIn("parallel index in Y.domain", definition.pseudocode_en)
        self.assertIn("Y[index] = max(X[index], 0)", definition.pseudocode_en)
        self.assertIn("算子 Relu", definition.pseudocode_zh)
        self.assertIn("并行遍历 Y.domain 中的 index", definition.pseudocode_zh)
        self.assertIn("Y[index] = 最大值(X[index], 0)", definition.pseudocode_zh)

    def test_add_describes_broadcast_shape_and_indices(self) -> None:
        definition = lower_operator("", "Add", 18, {})

        self.assertIn(
            "constraint Y.shape == broadcast_shape(A.shape, B.shape)",
            definition.pseudocode_en,
        )
        self.assertIn(
            "a_index = broadcast_index(index, A.shape, Y.shape)",
            definition.pseudocode_en,
        )
        self.assertIn("约束 Y.shape == 广播形状(A.shape, B.shape)", definition.pseudocode_zh)

    def test_does_not_apply_modern_broadcasting_to_legacy_add(self) -> None:
        definition = lower_operator("", "Add", 6, {"broadcast": 1, "axis": 1})

        self.assertEqual(definition.status, "unsupported")
        self.assertEqual(definition.category, "UNDEFINED")

    def test_cast_names_the_target_type_in_both_languages(self) -> None:
        definition = lower_operator(
            "",
            "Cast",
            19,
            {"to": TensorProto.FLOAT, "saturate": 0},
        )

        self.assertEqual(definition.status, "supported")
        self.assertEqual(definition.category, "CONVERSION")
        self.assertIn("output Y: Tensor<FLOAT>", definition.pseudocode_en)
        self.assertIn("attribute to = FLOAT", definition.pseudocode_en)
        self.assertNotIn("saturate", definition.pseudocode_en)
        self.assertIn("Y[index] = cast(X[index], to)", definition.pseudocode_en)
        self.assertIn("输出 Y: Tensor<FLOAT>", definition.pseudocode_zh)
        self.assertIn("Y[index] = 类型转换(X[index], to)", definition.pseudocode_zh)

    def test_cast_preserves_float8_saturation_and_rounding_controls(self) -> None:
        definition = lower_operator(
            "",
            "Cast",
            24,
            {
                "to": TensorProto.FLOAT8E8M0,
                "saturate": 0,
                "round_mode": "nearest",
            },
        )

        self.assertIn("attribute to = FLOAT8E8M0", definition.pseudocode_en)
        self.assertIn("attribute saturate = false", definition.pseudocode_en)
        self.assertIn("attribute round_mode = nearest", definition.pseudocode_en)
        self.assertIn(
            "cast(X[index], to, saturate, round_mode)",
            definition.pseudocode_en,
        )

    def test_reshape_is_a_view_and_preserves_allowzero(self) -> None:
        definition = lower_operator("", "Reshape", 18, {"allowzero": 1})

        self.assertEqual(definition.category, "VIEW")
        self.assertIn("attribute allowzero = 1", definition.pseudocode_en)
        self.assertIn("view Y = reshape_view(X, shape)", definition.pseudocode_en)
        self.assertIn("视图 Y = reshape_view(X, shape)", definition.pseudocode_zh)

    def test_transpose_uses_onnx_perm_attribute(self) -> None:
        definition = lower_operator("", "Transpose", 18, {"perm": [0, 2, 1]})

        self.assertEqual(definition.category, "LAYOUT")
        self.assertIn("attribute perm = [0, 2, 1]", definition.pseudocode_en)
        self.assertIn("X[inverse_permute(output_index, perm)]", definition.pseudocode_en)
        self.assertIn("X[逆置换(output_index, perm)]", definition.pseudocode_zh)

    def test_unsupported_operator_has_an_explicit_bilingual_diagnostic(self) -> None:
        definition = lower_operator("com.example", "Mystery", 1, {})

        self.assertEqual(definition.status, "unsupported")
        self.assertEqual(definition.category, "UNDEFINED")
        self.assertEqual(definition.pseudocode_en, "")
        self.assertIn("not been defined", definition.diagnostic_en)
        self.assertIn("尚未定义", definition.diagnostic_zh)

    def test_foundational_shape_and_arithmetic_operators_have_semantics(self) -> None:
        div = lower_operator("", "Div", 18, {})
        shape = lower_operator("", "Shape", 18, {"start": 1, "end": 3})
        concat = lower_operator("", "Concat", 18, {"axis": -1})

        self.assertEqual(div.status, "supported")
        self.assertIn("Y[index] = A[a_index] / B[b_index]", div.pseudocode_en)
        self.assertEqual(shape.category, "SHAPE")
        self.assertIn("Y = 获取形状(X, start, end)", shape.pseudocode_zh)
        self.assertEqual(concat.category, "LAYOUT")
        self.assertIn("属性 axis = -1", concat.pseudocode_zh)
        self.assertIn("Y = 拼接(inputs, axis)", concat.pseudocode_zh)

    def test_constant_semantics_summarize_tensor_without_embedding_values(self) -> None:
        definition = lower_operator(
            "",
            "Constant",
            18,
            {"value": {"data_type": "FLOAT", "shape": [2, 4]}},
        )

        self.assertEqual(definition.status, "supported")
        self.assertEqual(definition.category, "CONSTANT")
        self.assertIn("Tensor<FLOAT>[2, 4]", definition.pseudocode_en)

    def test_foundational_registry_covers_common_onnx_families(self) -> None:
        operators = (
            "Constant",
            "Shape",
            "Unsqueeze",
            "Squeeze",
            "Expand",
            "Concat",
            "Slice",
            "Gather",
            "GatherElements",
            "ScatterElements",
            "Split",
            "Pad",
            "Tile",
            "Range",
            "Sub",
            "Div",
            "Pow",
            "Sqrt",
            "Exp",
            "Log",
            "Neg",
            "Equal",
            "Less",
            "Greater",
            "Where",
            "ReduceSum",
            "ReduceMean",
            "ReduceL2",
            "CumSum",
            "ArgMax",
            "TopK",
            "Softmax",
            "MatMul",
            "MatMulInteger",
            "Gemm",
            "Conv",
            "ConvTranspose",
            "BatchNormalization",
            "InstanceNormalization",
            "LayerNormalization",
            "DynamicQuantizeLinear",
            "Resize",
            "CastLike",
        )

        unsupported = [
            op_type
            for op_type in operators
            if lower_operator("", op_type, 18, {}).status != "supported"
        ]

        self.assertEqual(unsupported, [])

    def test_resolver_binds_small_constant_inputs(self) -> None:
        definition = SemanticResolver().resolve(
            NodeContext(
                domain="",
                op_type="Unsqueeze",
                opset_version=18,
                attributes={},
                inputs=(
                    TensorSpec("X", "FLOAT", ("2", "4")),
                    TensorSpec("axes_value", "INT64", ("1",)),
                ),
                outputs=(TensorSpec("Y", "FLOAT", ("2", "1", "4")),),
                constant_inputs={"axes_value": (1,)},
            )
        )

        self.assertEqual(definition.source, "registry")
        self.assertEqual(definition.confidence, "EXACT_RULE")
        self.assertIn("input  X: Tensor<FLOAT>", definition.pseudocode_en)
        self.assertIn("known input axes = [1]", definition.pseudocode_en)
        self.assertIn("已知输入 axes = [1]", definition.pseudocode_zh)

    def test_resolver_uses_schema_function_for_unregistered_operator(self) -> None:
        definition = SemanticResolver().resolve(
            NodeContext(
                domain="",
                op_type="MeanVarianceNormalization",
                opset_version=18,
                attributes={"axes": [0, 2, 3]},
                inputs=(TensorSpec("X", "FLOAT", ("1", "4", "8", "8")),),
                outputs=(TensorSpec("Y", "FLOAT", ("1", "4", "8", "8")),),
            )
        )

        self.assertEqual(definition.status, "supported")
        self.assertEqual(definition.source, "schema_function")
        self.assertEqual(definition.confidence, "EXACT_FUNCTION")
        self.assertEqual(definition.category, "COMPOSITE")
        self.assertIn("attribute axes = [0, 2, 3]", definition.pseudocode_en)
        self.assertIn("ReduceMean(", definition.pseudocode_en)


if __name__ == "__main__":
    unittest.main()
