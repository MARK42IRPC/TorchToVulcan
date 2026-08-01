from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from jsonschema import Draft202012Validator

from torch_to_vulcan.compiler import (
    ExecutablePackageBuilder,
    ExecutablePackageError,
    load_executable_manifest,
    validate_executable_package,
)
from torch_to_vulcan.compiler.vulkan import VerificationRunner, detect_toolchain
from torch_to_vulcan.compiler.vulkan.kernels import (
    KernelContext,
    KernelTensor,
    default_kernel_registry,
)
from torch_to_vulcan.cli import main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_SPIRV = b"\x03\x02\x23\x07" + b"\x00" * 16


def add_plan():
    return default_kernel_registry().select(
        KernelContext(
            "",
            "Add",
            18,
            {},
            (
                KernelTensor("a", "FLOAT", ("1", "4")),
                KernelTensor("b", "FLOAT", ("1", "4")),
            ),
            (KernelTensor("y", "FLOAT", ("1", "4")),),
        )
    )


def package_builder() -> ExecutablePackageBuilder:
    builder = ExecutablePackageBuilder("persisted-add")
    builder.add_tensor("a", "FLOAT", (1, 4), storage="external")
    builder.add_constant("b", "FLOAT", (1,), b"\x00\x00\x80?")
    builder.add_constant("c", "FLOAT", (1,), b"\x00\x00\x00@")
    builder.add_tensor("y", "FLOAT", (1, 4), storage="external")
    builder.add_tensor("y2", "FLOAT", (1, 4), storage="transient")
    builder.bind_input("a")
    builder.bind_output("y")
    certificate = {
        "target_id": "add-fp32",
        "semantic_key": "ai.onnx::Add@18:test",
        "kernel_id": "elementwise.add.fp32",
        "status": "DEVICE_VERIFIED",
        "cases_passed": 3,
        "cases_total": 3,
    }
    step = add_plan().steps[0]
    builder.add_dispatch(
        "add_0",
        "elementwise.add.fp32",
        step,
        TEST_SPIRV,
        ("a", "b", "y"),
        certificate=certificate,
    )
    builder.add_dispatch(
        "add_1",
        "elementwise.add.fp32",
        step,
        TEST_SPIRV,
        ("y", "c", "y2"),
        certificate=certificate,
    )
    return builder


class ExecutablePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_writes_deduplicated_static_package_and_reloads_it(self) -> None:
        destination = self.root / "model.ttv"

        written = package_builder().write(destination, metadata={"source": "unit-test"})
        loaded = load_executable_manifest(destination)
        validated = validate_executable_package(destination)

        self.assertEqual(written, loaded)
        self.assertEqual(validated, loaded)
        self.assertEqual(loaded["format_version"], "0.1")
        self.assertEqual(len(loaded["shaders"]), 1)
        self.assertEqual(len(loaded["pipelines"]), 1)
        self.assertEqual(len(loaded["programs"][0]["steps"]), 2)
        self.assertEqual(loaded["certificate_store"]["count"], 1)
        self.assertEqual(loaded["tensors"]["b"]["storage"]["offset"], 0)
        self.assertEqual(loaded["tensors"]["c"]["storage"]["offset"], 256)
        self.assertEqual((destination / "constants" / "weights.bin").stat().st_size, 260)
        schema = json.loads(
            (REPOSITORY_ROOT / "schemas" / "model-package.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema).validate(loaded)

    def test_rejects_a_tampered_shader(self) -> None:
        destination = self.root / "model.ttv"
        manifest = package_builder().write(destination)
        shader_path = destination / manifest["shaders"][0]["file"]
        shader_path.write_bytes(TEST_SPIRV + b"\x00\x00\x00\x00")

        with self.assertRaisesRegex(ExecutablePackageError, "hash mismatch"):
            validate_executable_package(destination)

    def test_refuses_to_overwrite_an_existing_destination(self) -> None:
        destination = self.root / "model.ttv"
        destination.mkdir()

        with self.assertRaisesRegex(ExecutablePackageError, "already exists"):
            package_builder().write(destination)

    def test_rejects_constant_bytes_that_do_not_match_tensor_shape(self) -> None:
        builder = ExecutablePackageBuilder("invalid-constant")

        with self.assertRaisesRegex(ExecutablePackageError, "expected 16"):
            builder.add_constant("weight", "FLOAT", (4,), b"\x00" * 4)

    def test_cli_validates_a_materialized_package(self) -> None:
        destination = self.root / "model.ttv"
        package_builder().write(destination)
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(["validate-package", str(destination)])

        self.assertEqual(exit_code, 0)
        self.assertIn("valid TTV executable package 0.1", output.getvalue())
        self.assertIn("2 dispatches, 1 shaders", output.getvalue())

    @unittest.skipUnless(
        detect_toolchain().glslang_validator,
        "a GLSL compiler is required",
    )
    def test_persists_a_real_compiled_spirv_module(self) -> None:
        plan = add_plan()
        spirv, stage, message = VerificationRunner()._compile_shader(
            plan.steps[0].shader.source
        )
        self.assertIsNotNone(spirv, message)
        self.assertIn(stage, {"SPIRV_COMPILED", "SPIRV_VALIDATED"})
        builder = ExecutablePackageBuilder("compiled-add")
        for tensor_id in ("a", "b", "y"):
            builder.add_tensor(tensor_id, "FLOAT", (1, 4), storage="external")
        builder.bind_input("a")
        builder.bind_input("b")
        builder.bind_output("y")
        builder.add_dispatch(
            "add_0",
            plan.kernel_id,
            plan.steps[0],
            spirv or b"",
            ("a", "b", "y"),
        )

        destination = self.root / "compiled.ttv"
        manifest = builder.write(destination)

        self.assertEqual(len(manifest["shaders"]), 1)
        self.assertGreater((destination / manifest["shaders"][0]["file"]).stat().st_size, 20)
        validate_executable_package(destination)


if __name__ == "__main__":
    unittest.main()
