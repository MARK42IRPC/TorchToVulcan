from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from torch_to_vulcan.compiler import compile_static_model
from torch_to_vulcan.compiler.vulkan.runtime import VulkanPackageRuntime
from torch_to_vulcan.compiler.vulkan.verify import detect_toolchain


def make_chain_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "bias"], ["z"], name="add"),
            helper.make_node("Relu", ["z"], ["y"], name="relu"),
        ],
        "runtime_chain",
        [x],
        [y],
        [bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def make_identity_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        "runtime_identity",
        [x],
        [y],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def make_constant_model():
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    value = helper.make_tensor("value", TensorProto.FLOAT, [1, 4], [1.0, -2.0, 3.0, 4.0])
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["y"], name="constant", value=value)],
        "runtime_constant",
        [],
        [y],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def make_matmul_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [3, 2],
        [1.0, 2.0, 0.0, 1.0, -1.0, 0.5],
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="matmul")],
        "runtime_matmul",
        [x],
        [y],
        [weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


@unittest.skipUnless(
    detect_toolchain().vulkaninfo
    and detect_toolchain().glslang_validator
    and detect_toolchain().vulkan_binding,
    "Vulkan device, Python binding, and shader compiler are required",
)
class VulkanPackageRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.destination = Path(self.temporary_directory.name) / "chain.ttv"
        compile_static_model(make_chain_model(), self.destination)

    def test_executes_a_linear_program_and_reuses_resources(self) -> None:
        inputs = {"x": np.asarray([[-2.0, 0.0, 2.0, 3.0]], dtype=np.float32)}
        with VulkanPackageRuntime(self.destination) as runtime:
            first = runtime.run(inputs)
            second = runtime.run(inputs)

        np.testing.assert_allclose(first.outputs["y"], [[0.0, 1.0, 3.0, 4.0]])
        np.testing.assert_allclose(second.outputs["y"], first.outputs["y"])
        self.assertEqual(first.device_name, second.device_name)
        self.assertGreater(first.elapsed_ms, 0.0)

    def test_executes_the_linear_program_with_device_local_staging(self) -> None:
        inputs = {"x": np.asarray([[-2.0, 0.0, 2.0, 3.0]], dtype=np.float32)}
        with VulkanPackageRuntime(self.destination, device_local=True) as runtime:
            first = runtime.run(inputs)
            resident = runtime.run(None, transfer_inputs=False, read_outputs=False)
            second = runtime.run(None, transfer_inputs=False, read_outputs=True)

            self.assertIn(runtime.memory_mode, {"device-local", "host-visible"})

        np.testing.assert_allclose(first.outputs["y"], [[0.0, 1.0, 3.0, 4.0]])
        self.assertEqual(resident.outputs, {})
        np.testing.assert_allclose(second.outputs["y"], first.outputs["y"])

    def test_rejects_input_shape_or_dtype_changes_after_prepare(self) -> None:
        with VulkanPackageRuntime(self.destination) as runtime:
            runtime.run({"x": np.zeros((1, 4), dtype=np.float32)})
            with self.assertRaisesRegex(RuntimeError, "dtype"):
                runtime.run({"x": np.zeros((1, 4), dtype=np.float16)})
            with self.assertRaisesRegex(RuntimeError, "shape"):
                runtime.run({"x": np.zeros((4,), dtype=np.float32)})

    def test_device_local_copy_only_programs(self) -> None:
        cases = (
            (make_identity_model(), {"x": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)}, [[1.0, 2.0, 3.0, 4.0]]),
            (make_constant_model(), {}, [[1.0, -2.0, 3.0, 4.0]]),
        )
        for model, inputs, expected in cases:
            with self.subTest(model=model.graph.name):
                destination = Path(self.temporary_directory.name) / f"{model.graph.name}.ttv"
                compile_static_model(model, destination)
                with VulkanPackageRuntime(destination, device_local=True) as runtime:
                    result = runtime.run(inputs)
                    self.assertEqual(runtime.memory_mode, "device-local")
                np.testing.assert_allclose(result.outputs["y"], expected)

    def test_benchmark_reports_persistent_dispatch_latency(self) -> None:
        inputs = {"x": np.ones((1, 4), dtype=np.float32)}
        with VulkanPackageRuntime(self.destination) as runtime:
            benchmark = runtime.benchmark(inputs, warmup=1, iterations=2)

        self.assertEqual(benchmark.iterations, 2)
        self.assertEqual(benchmark.warmup, 1)
        self.assertGreaterEqual(benchmark.min_ms, 0.0)
        self.assertLessEqual(benchmark.min_ms, benchmark.mean_ms)
        self.assertLessEqual(benchmark.mean_ms, benchmark.max_ms)

    def test_executes_matmul_with_constant_weight(self) -> None:
        destination = Path(self.temporary_directory.name) / "matmul.ttv"
        compile_static_model(make_matmul_model(), destination)
        inputs = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float32)
        with VulkanPackageRuntime(destination) as runtime:
            result = runtime.run({"x": inputs})

        expected = inputs @ np.asarray(
            [[1.0, 2.0], [0.0, 1.0], [-1.0, 0.5]], dtype=np.float32
        )
        np.testing.assert_allclose(result.outputs["y"], expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
