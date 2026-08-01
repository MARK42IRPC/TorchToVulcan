from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from torch_to_vulcan.compiler import compile_static_model
from torch_to_vulcan.compiler import ExecutablePackageBuilder
from torch_to_vulcan.compiler.vulkan.runtime import VulkanPackageRuntime
from torch_to_vulcan.compiler.vulkan.verify import detect_toolchain
from torch_to_vulcan.compiler.vulkan.kernels import KernelContext, KernelTensor, default_kernel_registry


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


def make_batched_matmul_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 2])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [1, 4, 2],
        [1.0, 2.0, 0.0, 1.0, -1.0, 0.5, 2.0, -1.0],
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="batched_matmul")],
        "runtime_batched_matmul",
        [x],
        [y],
        [weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def make_reduce_mean_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["y"], name="reduce_mean", keepdims=0)],
        "runtime_reduce_mean",
        [x],
        [y],
        [axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def make_softmax_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["x"], ["y"], name="softmax", axis=1)],
        "runtime_softmax",
        [x],
        [y],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def make_layer_normalization_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [4], [1.0, 0.5, 2.0, -1.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [4], [0.0, 1.0, -1.0, 0.5])
    graph = helper.make_graph(
        [
            helper.make_node(
                "LayerNormalization",
                ["x", "scale", "bias"],
                ["y"],
                name="layer_norm",
                axis=-1,
                epsilon=1e-5,
            )
        ],
        "runtime_layer_normalization",
        [x],
        [y],
        [scale, bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_transformer_block_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 2])
    projection = helper.make_tensor_value_info("projection", TensorProto.FLOAT, [2, 3, 4])
    biased = helper.make_tensor_value_info("biased", TensorProto.FLOAT, [2, 3, 4])
    normalized = helper.make_tensor_value_info("normalized", TensorProto.FLOAT, [2, 3, 4])
    probabilities = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [2, 3, 4])
    projection_weight = helper.make_tensor(
        "projection_weight",
        TensorProto.FLOAT,
        [4, 4],
        [1.0, 0.0, 0.5, -1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 1.0, 0.0, 2.0, 1.0, -1.0, 0.0, 0.5],
    )
    projection_bias = helper.make_tensor(
        "projection_bias", TensorProto.FLOAT, [4], [0.25, -0.5, 1.0, 0.0]
    )
    norm_scale = helper.make_tensor(
        "norm_scale", TensorProto.FLOAT, [4], [1.0, 0.5, 2.0, -1.0]
    )
    norm_bias = helper.make_tensor(
        "norm_bias", TensorProto.FLOAT, [4], [0.0, 1.0, -1.0, 0.5]
    )
    output_weight = helper.make_tensor(
        "output_weight",
        TensorProto.FLOAT,
        [4, 2],
        [1.0, -1.0, 0.0, 0.5, 2.0, 1.0, -0.5, 2.0],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul",
                ["x", "projection_weight"],
                ["projection"],
                name="projection",
            ),
            helper.make_node(
                "Add",
                ["projection", "projection_bias"],
                ["biased"],
                name="projection_bias_add",
            ),
            helper.make_node(
                "LayerNormalization",
                ["biased", "norm_scale", "norm_bias"],
                ["normalized"],
                name="layer_norm",
                axis=-1,
                epsilon=1e-5,
            ),
            helper.make_node(
                "Softmax",
                ["normalized"],
                ["probabilities"],
                name="attention_softmax",
                axis=-1,
            ),
            helper.make_node(
                "MatMul",
                ["probabilities", "output_weight"],
                ["y"],
                name="output_projection",
            ),
        ],
        "runtime_transformer_block",
        [x],
        [y],
        [projection_weight, projection_bias, norm_scale, norm_bias, output_weight],
        value_info=[projection, biased, normalized, probabilities],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_stateful_add_package(destination: Path) -> None:
    plan = default_kernel_registry().select(
        KernelContext(
            "",
            "Add",
            18,
            {},
            (
                KernelTensor("cache", "FLOAT", ("1", "4")),
                KernelTensor("x", "FLOAT", ("1", "4")),
            ),
            (KernelTensor("cache", "FLOAT", ("1", "4")),),
        )
    )
    from torch_to_vulcan.compiler.vulkan.verify import VerificationRunner

    spirv, _stage, message = VerificationRunner()._compile_shader(plan.steps[0].shader.source)
    if spirv is None:
        raise RuntimeError(message)
    builder = ExecutablePackageBuilder("stateful-add")
    builder.add_tensor("x", "FLOAT", (1, 4), storage="external")
    builder.add_state_tensor(
        "cache",
        "FLOAT",
        (1, 4),
        state_id="kv_cache",
        update_program="decode",
    )
    builder.add_subprogram("decode", inputs=("x", "cache"), outputs=("cache",))
    builder.add_host_loop(
        "decode_loop",
        "decode",
        max_iterations=2,
        stop_tensor="cache",
    )
    builder.add_dispatch(
        "decode_add",
        plan.kernel_id,
        plan.steps[0],
        spirv,
        ("cache", "x", "cache"),
        program_id="decode",
    )
    builder.write(destination)


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

    def test_executes_batched_matmul_with_broadcast_constant_weight(self) -> None:
        destination = Path(self.temporary_directory.name) / "batched-matmul.ttv"
        compile_static_model(make_batched_matmul_model(), destination)
        inputs = np.asarray(
            [
                [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 1.0], [2.0, -1.0, 0.0, 3.0]],
                [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [-2.0, 0.0, 1.0, 2.0]],
            ],
            dtype=np.float32,
        )
        weight = np.asarray(
            [[1.0, 2.0], [0.0, 1.0], [-1.0, 0.5], [2.0, -1.0]],
            dtype=np.float32,
        )
        with VulkanPackageRuntime(destination) as runtime:
            result = runtime.run({"x": inputs})

        expected = np.matmul(inputs, weight)
        np.testing.assert_allclose(result.outputs["y"], expected, rtol=1e-5, atol=1e-6)

    def test_executes_reduce_mean_softmax_and_layer_normalization(self) -> None:
        cases = (
            (
                make_reduce_mean_model(),
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                lambda value: np.mean(value, axis=1),
            ),
            (
                make_softmax_model(),
                np.asarray(
                    [[[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 1.0, 2.0], [2.0, 0.0, -2.0, 1.0]],
                     [[-1.0, 0.5, 2.0, 3.0], [1.0, 2.0, -1.0, 0.0], [0.0, -2.0, 1.0, 2.0]]],
                    dtype=np.float32,
                ),
                lambda value: np.exp(value - np.max(value, axis=1, keepdims=True))
                / np.sum(np.exp(value - np.max(value, axis=1, keepdims=True)), axis=1, keepdims=True),
            ),
            (
                make_layer_normalization_model(),
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                lambda value: (
                    (value - np.mean(value, axis=-1, keepdims=True))
                    / np.sqrt(np.var(value, axis=-1, keepdims=True) + 1e-5)
                )
                * np.asarray([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
                + np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32),
            ),
        )
        for model, inputs, expected_fn in cases:
            with self.subTest(model=model.graph.name):
                destination = Path(self.temporary_directory.name) / f"{model.graph.name}.ttv"
                compile_static_model(model, destination)
                with VulkanPackageRuntime(destination) as runtime:
                    result = runtime.run({"x": inputs})
                np.testing.assert_allclose(
                    result.outputs["y"], expected_fn(inputs), rtol=2e-4, atol=2e-5
                )

    def test_executes_a_static_transformer_block(self) -> None:
        import onnxruntime as ort

        destination = Path(self.temporary_directory.name) / "transformer-block.ttv"
        model = make_transformer_block_model()
        report = compile_static_model(model, destination)
        self.assertEqual(report.dispatches, 5)

        inputs = np.asarray(
            [
                [[1.0, -2.0, 0.5, 3.0], [0.0, 1.0, -1.0, 2.0], [2.0, 0.5, 1.0, -0.5]],
                [[-1.0, 2.0, 0.0, 1.0], [3.0, -1.0, 2.0, 0.5], [0.5, 1.5, -2.0, 2.0]],
            ],
            dtype=np.float32,
        )
        projection_weight = np.asarray(
            [[1.0, 0.0, 0.5, -1.0], [0.0, 2.0, 1.0, 0.0], [-0.5, 1.0, 0.0, 2.0], [1.0, -1.0, 0.0, 0.5]],
            dtype=np.float32,
        )
        projection_bias = np.asarray([0.25, -0.5, 1.0, 0.0], dtype=np.float32)
        norm_scale = np.asarray([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
        norm_bias = np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        output_weight = np.asarray(
            [[1.0, -1.0], [0.0, 0.5], [2.0, 1.0], [-0.5, 2.0]],
            dtype=np.float32,
        )
        projection = np.matmul(inputs, projection_weight) + projection_bias
        normalized = (
            (projection - np.mean(projection, axis=-1, keepdims=True))
            / np.sqrt(np.var(projection, axis=-1, keepdims=True) + 1e-5)
        ) * norm_scale + norm_bias
        probabilities = np.exp(
            normalized - np.max(normalized, axis=-1, keepdims=True)
        )
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        expected = np.matmul(probabilities, output_weight)

        with VulkanPackageRuntime(destination) as runtime:
            result = runtime.run({"x": inputs})

        np.testing.assert_allclose(result.outputs["y"], expected, rtol=2e-4, atol=2e-5)
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        ort_output = session.run(["y"], {"x": inputs})[0]
        np.testing.assert_allclose(result.outputs["y"], ort_output, rtol=2e-4, atol=2e-5)

    def test_reuses_a_stateful_subprogram_and_resets_state(self) -> None:
        destination = Path(self.temporary_directory.name) / "stateful.ttv"
        make_stateful_add_package(destination)
        inputs = {"x": np.asarray([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)}

        with VulkanPackageRuntime(destination, device_local=True) as runtime:
            first = runtime.run_program("decode", inputs)
            runtime.run_program("decode", inputs, read_outputs=False)
            second = runtime.run_program("decode", inputs)
            runtime.reset_state("kv_cache")
            reset = runtime.run_program("decode", inputs)
            loop = runtime.run_loop("decode_loop", inputs)

        np.testing.assert_allclose(first.outputs["cache"], inputs["x"])
        np.testing.assert_allclose(second.outputs["cache"], inputs["x"] * 3.0)
        np.testing.assert_allclose(reset.outputs["cache"], inputs["x"])
        np.testing.assert_allclose(loop.outputs["cache"], inputs["x"] * 2.0)


if __name__ == "__main__":
    unittest.main()
