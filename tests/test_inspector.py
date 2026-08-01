from __future__ import annotations

import bz2
import gzip
import json
import lzma
import os
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import py7zr
import onnx
from onnx import TensorProto, helper

from torch_to_vulcan.cli import main
from torch_to_vulcan.importer import (
    InspectionError,
    MemoryConfirmationRequired,
    inspect_archive,
    inspect_path,
    source_format,
    supported_input_suffixes,
)


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


def make_transpose_chain_model() -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["input"], ["middle"], name="transpose_0", perm=[0, 2, 1]),
        helper.make_node("Transpose", ["middle"], ["output"], name="transpose_1", perm=[0, 2, 1]),
    ]
    graph = helper.make_graph(nodes, "transpose_chain", [input_info], [output_info])
    graph.value_info.append(
        helper.make_tensor_value_info("middle", TensorProto.FLOAT, [1, 3, 2])
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    ).SerializeToString()


def make_unannotated_relu_chain_model() -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])
    nodes = [
        helper.make_node("Relu", ["input"], ["middle"], name="relu_0"),
        helper.make_node("Relu", ["middle"], ["output"], name="relu_1"),
    ]
    graph = helper.make_graph(nodes, "relu_chain", [input_info], [output_info])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    ).SerializeToString()


def make_cast_model() -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.INT32, [1, 4])
    node = helper.make_node("Cast", ["input"], ["output"], name="cast_0", to=TensorProto.INT32)
    graph = helper.make_graph([node], "cast_graph", [input_info], [output_info])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    ).SerializeToString()


def make_local_function_model() -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Double", ["input"], ["output"], domain="local")
    graph = helper.make_graph([node], "local_function_graph", [input_info], [output_info])
    function = helper.make_function(
        "local",
        "Double",
        ["X"],
        ["Y"],
        [helper.make_node("Add", ["X", "X"], ["Y"])],
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("local", 1)],
    )
    model.functions.extend([function])
    return model.SerializeToString()


def make_unsqueeze_with_constant_axes_model() -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 1, 4])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    node = helper.make_node("Unsqueeze", ["input", "axes"], ["output"])
    graph = helper.make_graph(
        [node],
        "constant_axes_graph",
        [input_info],
        [output_info],
        initializer=[axes],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    ).SerializeToString()


class OnnxInspectorTests(unittest.TestCase):
    def temporary_path(self, filename: str) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        return Path(temporary_directory.name) / filename

    def write_archive(self, entries: dict[str, bytes]) -> Path:
        path = self.temporary_path("models.zip")
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
        path = self.temporary_path("relu.onnx")
        path.write_bytes(make_unary_model("Relu", graph_name="direct_relu"))

        report = inspect_path(path)

        self.assertEqual(report.source, str(path))
        self.assertEqual(report.source_type, "onnx")
        self.assertEqual(len(report.models), 1)
        self.assertEqual(report.models[0].path, "relu.onnx")
        self.assertEqual(report.operator_summary[0].op_type, "Relu")
        values = {value.name: value for value in report.models[0].graphs[0].values}
        self.assertEqual(values["input"].data_type, "FLOAT")
        self.assertEqual(values["input"].shape, ("1", "4"))
        self.assertEqual(values["output"].data_type, "FLOAT")

    def test_reports_attributes_opset_and_deduplicated_operator_semantics(self) -> None:
        path = self.temporary_path("transpose.onnx")
        path.write_bytes(make_transpose_chain_model())

        report = inspect_path(path)

        model = report.models[0]
        operators = model.graphs[0].operators
        self.assertEqual([operator.opset_version for operator in operators], [18, 18])
        self.assertEqual(operators[0].attributes[0].name, "perm")
        self.assertEqual(operators[0].attributes[0].kind, "INTS")
        self.assertEqual(operators[0].attributes[0].value, [0, 2, 1])
        self.assertEqual(operators[0].semantics_key, operators[1].semantics_key)
        self.assertEqual(len(model.semantics), 1)
        self.assertEqual(model.semantics[0].category, "LAYOUT")
        self.assertIn("attribute perm = [0, 2, 1]", model.semantics[0].pseudocode_en)
        self.assertIn("属性 perm = [0, 2, 1]", model.semantics[0].pseudocode_zh)

    def test_shape_inference_reports_unannotated_intermediate_values(self) -> None:
        path = self.temporary_path("relu-chain.onnx")
        path.write_bytes(make_unannotated_relu_chain_model())

        report = inspect_path(path)

        values = {value.name: value for value in report.models[0].graphs[0].values}
        self.assertEqual(values["middle"].data_type, "FLOAT")
        self.assertEqual(values["middle"].shape, ("batch", "4"))

    def test_shape_inference_failure_falls_back_to_original_model(self) -> None:
        path = self.temporary_path("relu.onnx")
        path.write_bytes(make_unary_model("Relu", graph_name="fallback"))

        with patch(
            "torch_to_vulcan.importer.inspector.onnx.shape_inference.infer_shapes",
            side_effect=onnx.shape_inference.InferenceError("unsupported custom schema"),
        ):
            report = inspect_path(path)

        self.assertEqual(len(report.models), 1)
        self.assertEqual(report.models[0].graph_name, "fallback")
        self.assertEqual(report.errors, [])

    def test_distinguishes_unknown_rank_from_a_scalar(self) -> None:
        from torch_to_vulcan.importer import inspector

        unknown_rank = inspector._value_info_report(
            helper.make_tensor_value_info("unknown", TensorProto.FLOAT, None)
        )
        scalar = inspector._value_info_report(
            helper.make_tensor_value_info("scalar", TensorProto.FLOAT, [])
        )

        self.assertFalse(unknown_rank.shape_known)
        self.assertTrue(scalar.shape_known)
        self.assertEqual(unknown_rank.shape, ())
        self.assertEqual(scalar.shape, ())

    def test_lowers_cast_attributes_into_readable_semantics(self) -> None:
        path = self.temporary_path("cast.onnx")
        path.write_bytes(make_cast_model())

        report = inspect_path(path)

        operator = report.models[0].graphs[0].operators[0]
        semantics = report.models[0].semantics[0]
        self.assertEqual(operator.op_type, "Cast")
        self.assertEqual(operator.attributes[0].name, "to")
        self.assertEqual(operator.attributes[0].value, TensorProto.INT32)
        self.assertEqual(semantics.status, "supported")
        self.assertEqual(semantics.category, "CONVERSION")
        self.assertIn("Tensor<INT32>", semantics.pseudocode_en)
        self.assertIn("类型转换(X[index], to)", semantics.pseudocode_zh)

    def test_expands_model_local_function_semantics(self) -> None:
        path = self.temporary_path("local-function.onnx")
        path.write_bytes(make_local_function_model())

        report = inspect_path(path)

        semantics = report.models[0].semantics[0]
        self.assertEqual(semantics.status, "supported")
        self.assertEqual(semantics.category, "COMPOSITE")
        self.assertEqual(semantics.source, "model_function")
        self.assertEqual(semantics.confidence, "EXACT_FUNCTION")
        self.assertIn("Y = Add(X, X)", semantics.pseudocode_en)

    def test_specializes_semantics_with_small_initializer_inputs(self) -> None:
        path = self.temporary_path("constant-axes.onnx")
        path.write_bytes(make_unsqueeze_with_constant_axes_model())

        report = inspect_path(path)

        semantics = report.models[0].semantics[0]
        self.assertIn("known input axes = [1]", semantics.pseudocode_en)
        self.assertIn("已知输入 axes = [1]", semantics.pseudocode_zh)

    def test_does_not_read_external_initializer_for_semantic_binding(self) -> None:
        from torch_to_vulcan.importer import inspector

        tensor = TensorProto(
            name="external_scalar",
            data_type=TensorProto.FLOAT,
            dims=[1],
            data_location=TensorProto.EXTERNAL,
        )
        location = tensor.external_data.add()
        location.key = "location"
        location.value = "weights.bin"

        self.assertIsNone(inspector._small_tensor_value(tensor, 16))

    def test_reports_tensor_values_for_each_nested_graph(self) -> None:
        path = self.temporary_path("conditional.onnx")
        path.write_bytes(make_if_model())

        report = inspect_path(path)

        values_by_graph = {
            graph.path: {value.name: value for value in graph.values}
            for graph in report.models[0].graphs
        }
        self.assertEqual(values_by_graph["conditional"]["condition"].data_type, "BOOL")
        then_values = next(
            values for path, values in values_by_graph.items() if path.endswith("then_branch")
        )
        self.assertEqual(then_values["then_y"].data_type, "FLOAT")
        self.assertEqual(then_values["then_y"].shape, ("1",))

    def test_requires_confirmation_above_memory_warning_threshold(self) -> None:
        path = self.temporary_path("large.onnx")
        path.write_bytes(make_unary_model("Relu", graph_name="large"))

        with patch(
            "torch_to_vulcan.importer.inspector.psutil.virtual_memory",
            return_value=SimpleNamespace(available=1),
        ):
            with self.assertRaises(MemoryConfirmationRequired):
                inspect_path(path)
            report = inspect_path(path, confirm_large_model=True)

        self.assertEqual(report.models[0].graph_name, "large")

    def test_inspects_tar_gz_archives(self) -> None:
        path = self.temporary_path("models.tar.gz")
        model = make_unary_model("Sigmoid", graph_name="tar_sigmoid")
        with tarfile.open(path, mode="w:gz") as archive:
            entry = tarfile.TarInfo("nested/sigmoid.onnx")
            entry.size = len(model)
            archive.addfile(entry, BytesIO(model))

        report = inspect_path(path)

        self.assertEqual(report.source_type, "tar")
        self.assertEqual(report.models[0].path, "nested/sigmoid.onnx")
        self.assertEqual(report.operator_summary[0].op_type, "Sigmoid")

    def test_inspects_7z_archives(self) -> None:
        path = self.temporary_path("models.7z")
        source = path.with_name("relu.onnx")
        source.write_bytes(make_unary_model("Relu", graph_name="seven_zip_relu"))
        with py7zr.SevenZipFile(path, mode="w") as archive:
            archive.write(source, arcname="models/relu.onnx")

        report = inspect_path(path)

        self.assertEqual(report.source_type, "7z")
        self.assertEqual(report.models[0].path, "models/relu.onnx")
        self.assertEqual(report.operator_summary[0].op_type, "Relu")

    def test_recognizes_rar_archives(self) -> None:
        path = self.temporary_path("models.RAR")

        self.assertIn(".rar", supported_input_suffixes())
        self.assertEqual(source_format(path), "rar")

    @unittest.skipUnless(os.name == "nt", "bundled UnRAR fixture targets Windows x64")
    def test_inspects_rar_archives_with_bundled_decoder(self) -> None:
        path = Path(__file__).parent / "fixtures" / "relu.rar"

        report = inspect_path(path)

        self.assertEqual(report.source_type, "rar")
        self.assertEqual(report.models[0].path, "live.onnx")
        self.assertEqual(report.operator_summary[0].op_type, "Relu")

    @unittest.skipUnless(os.name == "nt", "Windows process isolation behavior")
    def test_retries_unrar_after_a_console_control_interrupt(self) -> None:
        from torch_to_vulcan.importer import inspector

        interrupted = SimpleNamespace(returncode=0xC000013A, stderr=b"")
        completed = SimpleNamespace(returncode=0, stderr=b"")
        with patch.object(
            inspector.subprocess,
            "run",
            side_effect=[interrupted, completed],
        ) as run:
            inspector._extract_rar_models(Path("models.rar"), Path("output"))

        self.assertEqual(run.call_count, 2)
        self.assertTrue(
            run.call_args.kwargs["creationflags"] & inspector.subprocess.CREATE_NO_WINDOW
        )

    @unittest.skipUnless(os.name == "nt", "bundled UnRAR fixture targets Windows x64")
    def test_preserves_unrar_extraction_error_classification(self) -> None:
        path = Path(__file__).parent / "fixtures" / "relu.rar"
        failed = SimpleNamespace(returncode=2, stderr=b"decoder failure")

        with patch(
            "torch_to_vulcan.importer.inspector.subprocess.run",
            return_value=failed,
        ):
            with self.assertRaisesRegex(
                InspectionError,
                r"^UnRAR extraction failed with exit code 2",
            ):
                inspect_path(path)

    def test_inspects_compressed_onnx_files(self) -> None:
        model = make_unary_model("Neg", graph_name="compressed_neg")
        compressed_models = {
            "neg.onnx.gz": (gzip.compress(model), "gzip"),
            "neg.onnx.bz2": (bz2.compress(model), "bzip2"),
            "neg.onnx.xz": (lzma.compress(model), "xz"),
        }

        for filename, (content, source_type) in compressed_models.items():
            with self.subTest(filename=filename):
                path = self.temporary_path(filename)
                path.write_bytes(content)
                report = inspect_path(path)
                self.assertEqual(report.source_type, source_type)
                self.assertEqual(report.operator_summary[0].op_type, "Neg")

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
