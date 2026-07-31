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
