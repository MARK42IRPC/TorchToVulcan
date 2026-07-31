from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from onnx import TensorProto, helper

from torch_to_vulcan.cli import main
from torch_to_vulcan.importer import inspect_archive, inspect_path


def make_unary_model(op_type: str, *, graph_name: str) -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(op_type, ["input"], ["output"], name=f"{op_type.lower()}_0")
    graph = helper.make_graph([node], graph_name, [input_info], [output_info])
    model = helper.make_model(
        graph,
        producer_name="torch-to-vulcan-tests",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    return model.SerializeToString()


def make_if_model() -> bytes:
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

    then_graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["then_y"], name="then_relu")],
        "then_graph",
        [],
        [helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [1])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["else_y"], name="else_neg")],
        "else_graph",
        [],
        [helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [1])],
    )
    if_node = helper.make_node(
        "If",
        ["condition"],
        ["y"],
        name="choose",
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph([if_node], "conditional", [condition, x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()


class OnnxInspectorTests(unittest.TestCase):
    def write_archive(self, entries: dict[str, bytes]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "models.zip"
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
        return path

    def test_lists_all_models_and_operators(self) -> None:
        path = self.write_archive(
            {
                "models/relu.onnx": make_unary_model("Relu", graph_name="relu_graph"),
                "models/sigmoid.ONNX": make_unary_model("Sigmoid", graph_name="sigmoid_graph"),
                "README.txt": b"ignored",
            }
        )

        report = inspect_archive(path)

        self.assertEqual([model.path for model in report.models], [
            "models/relu.onnx",
            "models/sigmoid.ONNX",
        ])
        self.assertEqual(report.operator_count, 2)
        self.assertEqual(
            [(item.op_type, item.count) for item in report.operator_summary],
            [("Relu", 1), ("Sigmoid", 1)],
        )

    def test_includes_operators_in_nested_graphs(self) -> None:
        path = self.write_archive({"conditional.onnx": make_if_model()})

        report = inspect_archive(path)

        self.assertEqual(report.operator_count, 3)
        self.assertEqual(len(report.models[0].graphs), 3)
        self.assertEqual(report.models[0].graphs[0].inputs, ("condition", "x"))
        self.assertEqual(report.models[0].graphs[0].outputs, ("y",))
        self.assertEqual(
            {item.op_type for item in report.operator_summary},
            {"If", "Neg", "Relu"},
        )

    def test_keeps_valid_models_when_one_entry_is_malformed(self) -> None:
        path = self.write_archive(
            {
                "bad.onnx": b"not an ONNX protobuf",
                "valid.onnx": make_unary_model("Relu", graph_name="valid"),
            }
        )

        report = inspect_archive(path)

        self.assertEqual(len(report.models), 1)
        self.assertEqual(report.models[0].path, "valid.onnx")
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0].path, "bad.onnx")

    def test_inspects_a_direct_onnx_file(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "relu.onnx"
        path.write_bytes(make_unary_model("Relu", graph_name="direct_relu"))

        report = inspect_path(path)

        self.assertEqual(report.source, str(path))
        self.assertEqual(report.source_type, "onnx")
        self.assertEqual(len(report.models), 1)
        self.assertEqual(report.models[0].path, "relu.onnx")
        self.assertEqual(report.operator_summary[0].op_type, "Relu")

    def test_cli_json_output_is_machine_readable(self) -> None:
        path = self.write_archive({"relu.onnx": make_unary_model("Relu", graph_name="relu")})
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(["inspect", str(path), "--json"])

        self.assertEqual(exit_code, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["source_type"], "zip")
        self.assertEqual(value["operator_summary"][0]["op_type"], "Relu")


if __name__ == "__main__":
    unittest.main()
