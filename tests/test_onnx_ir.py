from __future__ import annotations

import unittest

import onnx
from onnx import TensorProto, helper

from torch_to_vulcan.compiler import ShapeProfile, normalize_onnx_model
from torch_to_vulcan.compiler.onnx import NormalizationError


class NormalizedOnnxIrTests(unittest.TestCase):
    def test_normalizes_tensor_contracts_constants_and_nodes(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2])
        weight = helper.make_tensor("weight", TensorProto.FLOAT, [4, 2], [0.0] * 8)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "weight"], ["y"], name="projection")],
            "projection",
            [x],
            [y],
            [weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        normalized = normalize_onnx_model(model)
        tensor = normalized.graph.tensors["x"]
        self.assertEqual(tensor.shape, ("batch", 4))
        self.assertEqual(tensor.layout, "contiguous")
        self.assertIsNone(tensor.strides)
        self.assertEqual(normalized.graph.tensors["weight"].strides, (2, 1))
        self.assertEqual(normalized.graph.nodes[0].op_type, "MatMul")
        self.assertEqual(normalized.graph.constants["weight"].data.__len__(), 32)

        specialized = normalized.specialize(ShapeProfile.from_mapping({"batch": 3}))
        self.assertEqual(specialized.graph.tensors["x"].shape, (3, 4))
        self.assertEqual(specialized.graph.tensors["x"].strides, (4, 1))

    def test_preserves_nested_graphs_as_normalized_graphs(self) -> None:
        condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
        then_graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["then_y"])],
            "then_branch",
            [],
            [helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [2])],
        )
        else_graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["else_y"])],
            "else_branch",
            [],
            [helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [2])],
        )
        node = helper.make_node(
            "If",
            ["condition"],
            ["y"],
            then_branch=then_graph,
            else_branch=else_graph,
        )
        model = helper.make_model(
            helper.make_graph([node], "conditional", [condition, x], [y]),
            opset_imports=[helper.make_opsetid("", 18)],
        )

        normalized = normalize_onnx_model(model)
        self.assertTrue(normalized.graph.nodes[0].has_subgraphs)
        self.assertEqual(
            {graph.name for graph in normalized.graph.nodes[0].subgraphs},
            {"then_branch", "else_branch"},
        )
        self.assertEqual(
            {tuple(graph.captures) for graph in normalized.graph.nodes[0].subgraphs},
            {("x",)},
        )

    def test_rejects_profile_that_does_not_bind_symbolic_dimension(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch"])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch"])
        model = helper.make_model(
            helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "identity", [x], [y]),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        normalized = normalize_onnx_model(model)
        with self.assertRaises(NormalizationError):
            normalized.specialize(ShapeProfile.from_mapping({"sequence": 4}))


if __name__ == "__main__":
    unittest.main()
