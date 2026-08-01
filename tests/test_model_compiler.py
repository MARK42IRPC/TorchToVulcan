from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from torch_to_vulcan.compiler import (
    ShapeProfile,
    StaticCompilationError,
    compile_static_model,
    validate_executable_package,
)
from torch_to_vulcan.compiler.vulkan.kernels import KernelContext, KernelTensor
from torch_to_vulcan.compiler.vulkan.verify import detect_toolchain

TEST_SPIRV = b"\x03\x02\x23\x07" + b"\x00" * 16


def make_add_model(shape: list[int | str] | None = None) -> onnx.ModelProto:
    input_shape = shape or [2, 3]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "bias"], ["y"], name="add_bias")],
        "static_add",
        [x],
        [y],
        [bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


class StaticModelCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_compiles_a_real_static_graph_contract_to_a_package(self) -> None:
        destination = self.root / "add.ttv"

        report = compile_static_model(
            make_add_model(),
            destination,
            shader_compiler=lambda _source: TEST_SPIRV,
        )

        manifest = validate_executable_package(destination)
        self.assertEqual(report.dispatches, 1)
        self.assertEqual(report.metadata_views, 0)
        self.assertEqual(manifest["bindings"], {"inputs": ["x"], "outputs": ["y"]})
        self.assertEqual(manifest["tensors"]["bias"]["storage"]["kind"], "constant")
        self.assertEqual(manifest["tensors"]["y"]["storage"]["kind"], "external")
        self.assertEqual(
            [item["tensor_id"] for item in manifest["programs"][0]["steps"][0]["resources"]],
            ["x", "bias", "y"],
        )

    @unittest.skipUnless(detect_toolchain().glslang_validator, "a GLSL compiler is required")
    def test_default_shader_compiler_persists_real_spirv(self) -> None:
        destination = self.root / "compiled-add.ttv"

        compile_static_model(make_add_model(), destination)

        manifest = validate_executable_package(destination)
        shader = destination / manifest["shaders"][0]["file"]
        self.assertGreater(shader.stat().st_size, len(TEST_SPIRV))

    def test_lowers_reshape_to_a_storage_view_without_a_shader(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
        shape = helper.make_tensor("new_shape", TensorProto.INT64, [2], [3, 2])
        graph = helper.make_graph(
            [helper.make_node("Reshape", ["x", "new_shape"], ["y"], name="reshape")],
            "reshape_view",
            [x],
            [y],
            [shape],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        destination = self.root / "reshape.ttv"

        report = compile_static_model(
            model,
            destination,
            shader_compiler=lambda _source: self.fail("view must not compile a shader"),
        )

        manifest = validate_executable_package(destination)
        self.assertEqual(report.dispatches, 0)
        self.assertEqual(report.metadata_views, 1)
        self.assertEqual(manifest["shaders"], [])
        self.assertEqual(manifest["tensors"]["y"]["storage"]["kind"], "view")
        self.assertEqual(manifest["tensors"]["y"]["storage"]["source_tensor"], "x")
        self.assertEqual(manifest["tensors"]["y"]["storage"]["strides"], [2, 1])

    def test_rejects_dynamic_shapes_without_writing_a_partial_package(self) -> None:
        destination = self.root / "dynamic.ttv"

        with self.assertRaises(StaticCompilationError) as raised:
            compile_static_model(
                make_add_model(["batch", 3]),
                destination,
                shader_compiler=lambda _source: TEST_SPIRV,
            )

        self.assertFalse(destination.exists())
        self.assertTrue(
            any(item.code == "DYNAMIC_SHAPE_UNSUPPORTED" for item in raised.exception.diagnostics)
        )

    def test_specializes_dynamic_shapes_with_a_profile(self) -> None:
        destination = self.root / "profiled.ttv"

        report = compile_static_model(
            make_add_model(["batch", 3]),
            destination,
            shape_profile=ShapeProfile.from_mapping({"batch": 2}, name="batch2"),
            shader_compiler=lambda _source: TEST_SPIRV,
        )

        manifest = validate_executable_package(destination)
        self.assertEqual(report.shape_profile.name, "batch2")
        self.assertEqual(manifest["tensors"]["x"]["shape"], [2, 3])
        self.assertEqual(manifest["metadata"]["shape_profile"], '{"dimensions": {"batch": 2}, "name": "batch2"}')

    def test_rejects_a_truly_unregistered_kernel_without_writing_a_package(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1])
        graph = helper.make_graph(
            [helper.make_node("UnknownOp", ["x"], ["y"], domain="com.example")],
            "unsupported",
            [x],
            [y],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.example", 1)],
        )
        destination = self.root / "unsupported.ttv"

        with self.assertRaises(StaticCompilationError) as raised:
            compile_static_model(
                model,
                destination,
                shader_compiler=lambda _source: TEST_SPIRV,
            )

        self.assertFalse(destination.exists())
        self.assertTrue(
            any(item.code == "KERNEL_UNSUPPORTED" for item in raised.exception.diagnostics)
        )

    def test_kernel_planning_uses_the_real_large_shape(self) -> None:
        from torch_to_vulcan.compiler.vulkan.kernels import default_kernel_registry

        element_count = 16_777_217
        plan = default_kernel_registry().select(
            KernelContext(
                "",
                "Relu",
                18,
                {},
                (KernelTensor("x", "FLOAT", (str(element_count),)),),
                (KernelTensor("y", "FLOAT", (str(element_count),)),),
            )
        )

        self.assertEqual(plan.steps[0].push_constants["element_count"], element_count)
        self.assertEqual(plan.steps[0].workgroups, (65_535, 2, 1))
        self.assertIn("gl_GlobalInvocationID.y", plan.steps[0].shader.source)

    def test_plans_two_dimensional_matmul(self) -> None:
        from torch_to_vulcan.compiler.vulkan.kernels import default_kernel_registry

        plan = default_kernel_registry().select(
            KernelContext(
                "",
                "MatMul",
                18,
                {},
                (
                    KernelTensor("x", "FLOAT", ("2", "3")),
                    KernelTensor("weight", "FLOAT", ("3", "4")),
                ),
                (KernelTensor("y", "FLOAT", ("2", "4")),),
            )
        )

        self.assertEqual(plan.kernel_id, "linear.matmul.fp32")
        self.assertEqual(plan.steps[0].workgroups, (1, 1, 1))
        self.assertIn("for (uint inner = 0u; inner < 3u; ++inner)", plan.steps[0].shader.source)
        self.assertIn("input_a[row * 3u + inner]", plan.steps[0].shader.source)

    def test_plans_gemm_transpose_and_bias_broadcast(self) -> None:
        from torch_to_vulcan.compiler.vulkan.kernels import default_kernel_registry

        plan = default_kernel_registry().select(
            KernelContext(
                "",
                "Gemm",
                18,
                {"transA": 1, "transB": 0, "alpha": 0.5, "beta": 2.0},
                (
                    KernelTensor("x", "FLOAT", ("3", "2")),
                    KernelTensor("weight", "FLOAT", ("3", "4")),
                    KernelTensor("bias", "FLOAT", ("4",)),
                ),
                (KernelTensor("y", "FLOAT", ("2", "4")),),
            )
        )

        self.assertEqual(plan.kernel_id, "linear.gemm.fp32")
        self.assertEqual(plan.steps[0].push_constants["element_count"], 8)
        self.assertIn("input_c[column]", plan.steps[0].shader.source)
        self.assertIn("0.5", plan.steps[0].shader.source)
        self.assertIn("2.0", plan.steps[0].shader.source)


if __name__ == "__main__":
    unittest.main()
