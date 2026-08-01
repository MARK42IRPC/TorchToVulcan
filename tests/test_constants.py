from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from torch_to_vulcan.compiler import (
    ExecutablePackageBuilder,
    fold_constant_subgraph,
    materialize_folded_constants,
    rewrite_model_with_folded_constants,
    validate_executable_package,
)


def make_shape_program(input_shape: list[int | str]):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
    result = helper.make_tensor_value_info("result", TensorProto.INT64, [3])
    index = helper.make_tensor("index_value", TensorProto.INT64, [], [1])
    axes = helper.make_tensor("axes_value", TensorProto.INT64, [1], [0])
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"], name="shape"),
        helper.make_node("Constant", [], ["index"], name="index", value=index),
        helper.make_node("Constant", [], ["axes"], name="axes", value=axes),
        helper.make_node("Gather", ["x_shape", "index"], ["selected"], axis=0),
        helper.make_node("Unsqueeze", ["selected", "axes"], ["selected_vector"]),
        helper.make_node(
            "Concat",
            ["x_shape", "selected_vector"],
            ["result"],
            axis=0,
        ),
    ]
    graph = helper.make_graph(nodes, "shape_program", [x], [result])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )


class ConstantFoldingTests(unittest.TestCase):
    def test_folds_a_static_shape_program_and_materializes_its_result(self) -> None:
        model = make_shape_program([2, 3])

        result = fold_constant_subgraph(model)

        self.assertEqual(len(result.folded_nodes), 6)
        self.assertEqual(result.runtime_node_indices, [])
        self.assertEqual(result.diagnostics, [])
        self.assertEqual(result.required_values, ["result"])
        np.testing.assert_array_equal(
            result.values["result"],
            np.asarray([2, 3, 3], dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "shape.ttv"
            builder = ExecutablePackageBuilder("static-shape-program")
            materialize_folded_constants(builder, result)
            builder.bind_output("result")
            manifest = builder.write(destination)

            validate_executable_package(destination)
            storage = manifest["tensors"]["result"]["storage"]
            self.assertEqual(storage["kind"], "constant")
            self.assertEqual(storage["length"], 24)
            self.assertEqual(
                (destination / "constants" / "weights.bin").read_bytes(),
                np.asarray([2, 3, 3], dtype=np.int64).tobytes(),
            )

    def test_dynamic_input_shape_stays_at_runtime(self) -> None:
        model = make_shape_program(["batch", 3])

        result = fold_constant_subgraph(model)

        self.assertEqual(
            [node.op_type for node in result.folded_nodes],
            ["Constant", "Constant"],
        )
        self.assertEqual(result.runtime_node_indices, [0, 3, 4, 5])
        self.assertEqual(result.required_values, ["index", "axes"])
        self.assertNotIn("result", result.values)

        rewritten = rewrite_model_with_folded_constants(model, result)
        onnx.checker.check_model(rewritten)
        self.assertEqual([node.op_type for node in rewritten.graph.node], [
            "Shape",
            "Gather",
            "Unsqueeze",
            "Concat",
        ])
        self.assertEqual(
            {initializer.name for initializer in rewritten.graph.initializer},
            {"index", "axes"},
        )
        output = ReferenceEvaluator(rewritten).run(
            None,
            {"x": np.zeros((5, 3), dtype=np.float32)},
        )[0]
        np.testing.assert_array_equal(output, np.asarray([5, 3, 3], dtype=np.int64))

    def test_rewrite_replaces_a_fully_folded_graph_with_an_initializer(self) -> None:
        model = make_shape_program([2, 3])
        result = fold_constant_subgraph(model)

        rewritten = rewrite_model_with_folded_constants(model, result)

        onnx.checker.check_model(rewritten)
        self.assertEqual(len(rewritten.graph.node), 0)
        self.assertEqual(
            [initializer.name for initializer in rewritten.graph.initializer],
            ["result"],
        )
        output = ReferenceEvaluator(rewritten).run(
            None,
            {"x": np.zeros((2, 3), dtype=np.float32)},
        )[0]
        np.testing.assert_array_equal(output, np.asarray([2, 3, 3], dtype=np.int64))

    def test_folds_size_from_a_static_runtime_tensor(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
        size = helper.make_tensor_value_info("size", TensorProto.INT64, [])
        graph = helper.make_graph(
            [helper.make_node("Size", ["x"], ["size"])],
            "static_size",
            [x],
            [size],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18)],
        )

        result = fold_constant_subgraph(model)

        self.assertEqual(result.runtime_node_indices, [])
        self.assertEqual(result.required_values, ["size"])
        self.assertEqual(result.values["size"].shape, ())
        self.assertEqual(result.values["size"].item(), 24)

    def test_constant_size_limit_is_a_diagnostic_not_a_crash(self) -> None:
        tensor = helper.make_tensor("large", TensorProto.FLOAT, [4], [1, 2, 3, 4])
        node = helper.make_node("Constant", [], ["large"], value=tensor)
        graph = helper.make_graph(
            [node],
            "large_constant",
            [],
            [helper.make_tensor_value_info("large", TensorProto.FLOAT, [4])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18)],
        )

        result = fold_constant_subgraph(model, max_constant_bytes=4)

        self.assertEqual(result.runtime_node_indices, [0])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertIn("limit is 4", result.diagnostics[0].message)

    def test_materializes_a_zero_length_constant_blob(self) -> None:
        result = fold_constant_subgraph(make_shape_program([0, 3]))
        result.values["empty"] = np.asarray([], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "empty.ttv"
            builder = ExecutablePackageBuilder("empty-constant")
            materialize_folded_constants(builder, result, names=("empty",))
            manifest = builder.write(destination)

            self.assertEqual(manifest["blobs"][0]["length"], 0)
            self.assertTrue((destination / "constants" / "weights.bin").is_file())
            validate_executable_package(destination)


if __name__ == "__main__":
    unittest.main()
