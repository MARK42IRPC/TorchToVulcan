from __future__ import annotations

import unittest
from unittest.mock import patch

from torch_to_vulcan.compiler.vulkan import (
    ToolchainCapabilities,
    VerificationRunner,
    VerificationTarget,
    VerificationTensor,
    default_kernel_registry,
)
from torch_to_vulcan.compiler.vulkan.kernels import KernelContext, KernelTensor
from torch_to_vulcan.compiler.vulkan.verify import detect_toolchain


def make_target(op_type: str = "Add") -> VerificationTarget:
    inputs = (
        VerificationTensor("a", "FLOAT", ("1", "256")),
        VerificationTensor("b", "FLOAT", ("1", "256")),
    )
    if op_type == "Relu":
        inputs = inputs[:1]
    return VerificationTarget(
        target_id=f"{op_type.lower()}-fp32",
        semantic_key=f"ai.onnx::{op_type}@18:test",
        domain="",
        op_type=op_type,
        opset_version=18,
        attributes={},
        inputs=inputs,
        outputs=(VerificationTensor("y", "FLOAT", ("1", "256")),),
    )


def make_cast_target(input_type: str, output_type: str, to: int) -> VerificationTarget:
    return VerificationTarget(
        target_id=f"cast-{input_type.lower()}-{output_type.lower()}",
        semantic_key="ai.onnx::Cast@18:test",
        domain="",
        op_type="Cast",
        opset_version=18,
        attributes={"to": to},
        inputs=(VerificationTensor("x", input_type, ("1", "256")),),
        outputs=(VerificationTensor("y", output_type, ("1", "256")),),
    )


def make_broadcast_target() -> VerificationTarget:
    return VerificationTarget(
        target_id="add-broadcast-fp32",
        semantic_key="ai.onnx::Add@18:broadcast",
        domain="",
        op_type="Add",
        opset_version=18,
        attributes={},
        inputs=(
            VerificationTensor("a", "FLOAT", ("2", "1", "4")),
            VerificationTensor("b", "FLOAT", ("1", "3", "4")),
        ),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "3", "4")),),
    )


def make_scalar_broadcast_target() -> VerificationTarget:
    return VerificationTarget(
        target_id="add-scalar-broadcast-fp32",
        semantic_key="ai.onnx::Add@18:scalar-broadcast",
        domain="",
        op_type="Add",
        opset_version=18,
        attributes={},
        inputs=(
            VerificationTensor("scalar", "FLOAT", ()),
            VerificationTensor("tensor", "FLOAT", ("2", "3", "4", "5")),
        ),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "3", "4", "5")),),
    )


def make_unary_target(
    op_type: str,
    attributes: dict[str, object] | None = None,
) -> VerificationTarget:
    return VerificationTarget(
        target_id=f"{op_type.lower()}-fp32",
        semantic_key=f"ai.onnx::{op_type}@18:unary",
        domain="",
        op_type=op_type,
        opset_version=18,
        attributes=attributes or {},
        inputs=(VerificationTensor("x", "FLOAT", ("2", "3", "4")),),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "3", "4")),),
    )


def make_transpose_target() -> VerificationTarget:
    return VerificationTarget(
        target_id="transpose-fp32",
        semantic_key="ai.onnx::Transpose@18:perm-021",
        domain="",
        op_type="Transpose",
        opset_version=18,
        attributes={"perm": [0, 2, 1]},
        inputs=(VerificationTensor("x", "FLOAT", ("2", "3", "4")),),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "4", "3")),),
    )


def make_concat_target() -> VerificationTarget:
    return VerificationTarget(
        target_id="concat-fp32",
        semantic_key="ai.onnx::Concat@18:axis-1",
        domain="",
        op_type="Concat",
        opset_version=18,
        attributes={"axis": 1},
        inputs=(
            VerificationTensor("a", "FLOAT", ("2", "1", "4")),
            VerificationTensor("b", "FLOAT", ("2", "2", "4")),
            VerificationTensor("c", "FLOAT", ("2", "3", "4")),
        ),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "6", "4")),),
    )


def make_matmul_target() -> VerificationTarget:
    return VerificationTarget(
        target_id="matmul-fp32",
        semantic_key="ai.onnx::MatMul@18:fp32-2d",
        domain="",
        op_type="MatMul",
        opset_version=18,
        attributes={},
        inputs=(
            VerificationTensor("a", "FLOAT", ("2", "3")),
            VerificationTensor("b", "FLOAT", ("3", "4")),
        ),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "4")),),
    )


def make_gemm_target() -> VerificationTarget:
    return VerificationTarget(
        target_id="gemm-fp32",
        semantic_key="ai.onnx::Gemm@18:fp32-bias",
        domain="",
        op_type="Gemm",
        opset_version=18,
        attributes={"transA": 1, "transB": 0, "alpha": 0.5, "beta": 2.0},
        inputs=(
            VerificationTensor("a", "FLOAT", ("3", "2")),
            VerificationTensor("b", "FLOAT", ("3", "4")),
            VerificationTensor("c", "FLOAT", ("4",)),
        ),
        outputs=(VerificationTensor("y", "FLOAT", ("2", "4")),),
    )


class VulkanKernelTests(unittest.TestCase):
    def test_generates_guarded_fp32_add_dispatch(self) -> None:
        target = make_target()
        plan = default_kernel_registry().select(
            KernelContext(
                domain=target.domain,
                op_type=target.op_type,
                opset_version=target.opset_version,
                attributes=target.attributes,
                inputs=tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.inputs
                ),
                outputs=tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.outputs
                ),
            )
        )

        self.assertEqual(plan.kernel_id, "elementwise.add.fp32")
        self.assertEqual(plan.steps[0].workgroups, (1, 1, 1))
        self.assertEqual(plan.steps[0].push_constants["element_count"], 256)
        self.assertIn("if (index >= params.element_count)", plan.steps[0].shader.source)
        self.assertIn("input_a[index] + input_b[index]", plan.steps[0].shader.source)

    def test_generates_scalar_to_rank_four_broadcast_indices(self) -> None:
        plan = default_kernel_registry().select(
            KernelContext(
                "",
                "Add",
                18,
                {},
                (
                    KernelTensor("scalar", "FLOAT", ()),
                    KernelTensor("tensor", "FLOAT", ("2", "3", "4", "5")),
                ),
                (KernelTensor("y", "FLOAT", ("2", "3", "4", "5")),),
            )
        )

        source = plan.steps[0].shader.source
        self.assertIn("input_a[0u] + input_b[index]", source)
        self.assertEqual(plan.steps[0].push_constants["element_count"], 120)

    def test_generates_multidirectional_broadcast_indices(self) -> None:
        target = make_broadcast_target()
        plan = default_kernel_registry().select(
            KernelContext(
                target.domain,
                target.op_type,
                target.opset_version,
                target.attributes,
                tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.inputs
                ),
                tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.outputs
                ),
            )
        )

        source = plan.steps[0].shader.source
        self.assertIn("((index / 12u) % 2u) * 4u", source)
        self.assertIn("((index / 4u) % 3u) * 4u", source)
        self.assertEqual(plan.steps[0].push_constants["element_count"], 24)

    def test_generates_transpose_permutation_indices(self) -> None:
        target = make_transpose_target()
        plan = default_kernel_registry().select(
            KernelContext(
                target.domain,
                target.op_type,
                target.opset_version,
                target.attributes,
                tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.inputs
                ),
                tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.outputs
                ),
            )
        )

        source = plan.steps[0].shader.source
        self.assertEqual(plan.kernel_id, "layout.transpose.fp32")
        self.assertIn("((index / 12u) % 2u) * 12u", source)
        self.assertIn("((index / 3u) % 4u)", source)
        self.assertIn("(index % 3u) * 4u", source)

    def test_generates_concat_segment_indices(self) -> None:
        target = make_concat_target()
        plan = default_kernel_registry().select(
            KernelContext(
                target.domain,
                target.op_type,
                target.opset_version,
                target.attributes,
                tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.inputs
                ),
                tuple(
                    KernelTensor(item.name, item.data_type, item.shape)
                    for item in target.outputs
                ),
            )
        )

        source = plan.steps[0].shader.source
        self.assertEqual(plan.kernel_id, "layout.concat.fp32")
        self.assertIn("uint axis_index = (index / 4u) % 6u", source)
        self.assertIn("if (axis_index < 1u)", source)
        self.assertIn("else if (axis_index < 3u)", source)
        self.assertIn("else if (axis_index < 6u)", source)
        self.assertEqual(plan.steps[0].push_constants["element_count"], 48)

    def test_reshape_creates_a_zero_dispatch_plan(self) -> None:
        plan = default_kernel_registry().select(
            KernelContext(
                "",
                "Reshape",
                18,
                {},
                (
                    KernelTensor("x", "FLOAT", ("2", "3")),
                    KernelTensor("shape", "INT64", ("2",)),
                ),
                (KernelTensor("y", "FLOAT", ("3", "2")),),
            )
        )

        self.assertTrue(plan.metadata_only)
        self.assertEqual(plan.dispatch_count, 0)

    def test_reshape_plan_receives_a_range_verified_certificate(self) -> None:
        target = VerificationTarget(
            target_id="reshape-view",
            semantic_key="ai.onnx::Reshape@18:test",
            domain="",
            op_type="Reshape",
            opset_version=18,
            attributes={},
            inputs=(
                VerificationTensor("x", "FLOAT", ("2", "3")),
                VerificationTensor("shape", "INT64", ("2",)),
            ),
            outputs=(VerificationTensor("y", "FLOAT", ("3", "2")),),
        )
        events: list[dict[str, object]] = []
        runner = VerificationRunner(
            capabilities=ToolchainCapabilities(
                vulkaninfo=None,
                glslang_validator=None,
                spirv_val=None,
                onnxruntime=False,
                vulkan_binding=False,
                executor_available=False,
                device_name="UNAVAILABLE",
            )
        )

        summary = runner.run([target], events.append)

        self.assertEqual(summary["verified"], 1)
        certificate = next(
            event["certificate"]
            for event in events
            if event["type"] == "certificate"
        )
        self.assertEqual(certificate["status"], "RANGE_VERIFIED")
        self.assertEqual(certificate["confidence"], 1.0)

    def test_unknown_reshape_is_blocked_instead_of_range_verified(self) -> None:
        target = VerificationTarget(
            target_id="reshape-unknown",
            semantic_key="ai.onnx::Reshape@18:unknown",
            domain="",
            op_type="Reshape",
            opset_version=18,
            attributes={},
            inputs=(
                VerificationTensor("x", "UNKNOWN", ()),
                VerificationTensor("shape", "INT64", ("1",)),
            ),
            outputs=(VerificationTensor("y", "UNKNOWN", ()),),
        )
        events: list[dict[str, object]] = []
        runner = VerificationRunner(
            capabilities=ToolchainCapabilities(
                vulkaninfo=None,
                glslang_validator=None,
                spirv_val=None,
                onnxruntime=False,
                vulkan_binding=False,
                executor_available=False,
                device_name="UNAVAILABLE",
            )
        )

        summary = runner.run([target], events.append)

        self.assertEqual(summary["blocked"], 1)
        certificate = next(
            event["certificate"]
            for event in events
            if event["type"] == "certificate"
        )
        self.assertEqual(certificate["status"], "BLOCKED")
        self.assertIn("数据类型", certificate["message"])

    def test_unknown_rank_is_blocked_instead_of_treated_as_a_scalar(self) -> None:
        target = VerificationTarget(
            target_id="relu-unknown-rank",
            semantic_key="ai.onnx::Relu@18:unknown-rank",
            domain="",
            op_type="Relu",
            opset_version=18,
            attributes={},
            inputs=(VerificationTensor("x", "FLOAT", (), False),),
            outputs=(VerificationTensor("y", "FLOAT", (), False),),
        )
        events: list[dict[str, object]] = []
        runner = VerificationRunner(
            capabilities=ToolchainCapabilities(
                vulkaninfo=None,
                glslang_validator=None,
                spirv_val=None,
                onnxruntime=False,
                vulkan_binding=False,
                executor_available=False,
                device_name="UNAVAILABLE",
            )
        )

        summary = runner.run([target], events.append)

        self.assertEqual(summary["blocked"], 1)
        certificate = next(
            event["certificate"]
            for event in events
            if event["type"] == "certificate"
        )
        self.assertIn("张量秩", certificate["message"])

    def test_missing_shader_compiler_produces_a_blocked_certificate_and_logs(self) -> None:
        events: list[dict[str, object]] = []
        runner = VerificationRunner(
            capabilities=ToolchainCapabilities(
                vulkaninfo="vulkaninfo",
                glslang_validator=None,
                spirv_val=None,
                onnxruntime=True,
                vulkan_binding=True,
                executor_available=False,
                device_name="Test GPU",
            )
        )

        summary = runner.run([make_target()], events.append)

        self.assertEqual(summary, {"total": 1, "verified": 0, "blocked": 1, "failed": 0})
        logs = [event for event in events if event["type"] == "log"]
        self.assertTrue(any("ONNX Runtime 参考输出完成" in str(event["message"]) for event in logs))
        self.assertTrue(any("glslangValidator" in str(event["message"]) for event in logs))
        progress = next(event for event in events if event["type"] == "progress")
        self.assertEqual((progress["current"], progress["total"]), (1, 1))
        certificate = next(event for event in events if event["type"] == "certificate")
        self.assertEqual(certificate["certificate"]["status"], "BLOCKED")
        self.assertEqual(
            certificate["certificate"]["last_completed"],
            "REFERENCE_EXECUTED",
        )
        self.assertEqual(certificate["certificate"]["cases_total"], 3)
        self.assertEqual(len(certificate["certificate"]["reference_hash"]), 64)

    def test_missing_vulkan_binding_blocks_after_spirv_compilation(self) -> None:
        events: list[dict[str, object]] = []
        runner = VerificationRunner(
            capabilities=ToolchainCapabilities(
                vulkaninfo=None,
                glslang_validator="test-compiler",
                spirv_val=None,
                onnxruntime=True,
                vulkan_binding=False,
                executor_available=False,
                device_name="UNAVAILABLE",
            )
        )

        with patch.object(
            runner,
            "_compile_shader",
            return_value=(b"\x03\x02\x23\x07", "SPIRV_COMPILED", "compiled"),
        ):
            summary = runner.run([make_target()], events.append)

        self.assertEqual(summary["blocked"], 1)
        certificate = next(
            event["certificate"]
            for event in events
            if event["type"] == "certificate"
        )
        self.assertEqual(certificate["status"], "BLOCKED")
        self.assertIn("Vulkan Python", certificate["message"])

    def test_reference_graph_uses_unique_names_for_reused_model_values(self) -> None:
        target = VerificationTarget(
            target_id="add-reused-name",
            semantic_key="ai.onnx::Add@18:reused-name",
            domain="",
            op_type="Add",
            opset_version=18,
            attributes={},
            inputs=(
                VerificationTensor("same", "FLOAT", ("2", "3")),
                VerificationTensor("same", "FLOAT", ("2", "3")),
            ),
            outputs=(VerificationTensor("same", "FLOAT", ("2", "3")),),
        )
        events: list[dict[str, object]] = []
        runner = VerificationRunner(
            capabilities=ToolchainCapabilities(
                vulkaninfo=None,
                glslang_validator=None,
                spirv_val=None,
                onnxruntime=True,
                vulkan_binding=False,
                executor_available=False,
                device_name="UNAVAILABLE",
            )
        )

        summary = runner.run([target], events.append)

        self.assertEqual(summary["blocked"], 1)
        certificate = next(
            event["certificate"]
            for event in events
            if event["type"] == "certificate"
        )
        self.assertEqual(certificate["last_completed"], "REFERENCE_EXECUTED")
        self.assertEqual(certificate["cases_total"], 3)

    @unittest.skipUnless(
        detect_toolchain().vulkaninfo and detect_toolchain().glslang_validator,
        "Vulkan device and shader compiler are required",
    )
    def test_device_verifies_initial_compute_catalog(self) -> None:
        events: list[dict[str, object]] = []
        targets = [
            make_target(op_type)
            for op_type in ("Add", "Mul", "Sub", "Div", "Relu")
        ] + [
            make_cast_target("FLOAT", "INT32", 6),
            make_cast_target("INT32", "FLOAT", 1),
            make_broadcast_target(),
            make_scalar_broadcast_target(),
            make_transpose_target(),
            make_concat_target(),
            make_matmul_target(),
            make_gemm_target(),
            *[
                make_unary_target(op_type)
                for op_type in (
                    "Neg",
                    "Exp",
                    "Floor",
                    "Sin",
                    "Cos",
                    "Sqrt",
                    "Tanh",
                    "Sigmoid",
                )
            ],
            make_unary_target("LeakyRelu", {"alpha": 0.1}),
            VerificationTarget(
                target_id="relu-dynamic-fp32",
                semantic_key="ai.onnx::Relu@18:dynamic",
                domain="",
                op_type="Relu",
                opset_version=18,
                attributes={},
                inputs=(
                    VerificationTensor("x", "FLOAT", ("token_count", "s0")),
                ),
                outputs=(
                    VerificationTensor("y", "FLOAT", ("token_count", "s0")),
                ),
            ),
        ]

        summary = VerificationRunner().run(targets, events.append)

        self.assertEqual(summary["verified"], len(targets))
        certificates = [
            event["certificate"]
            for event in events
            if event["type"] == "certificate"
        ]
        self.assertEqual(len(certificates), len(targets))
        self.assertTrue(
            all(item["status"] == "DEVICE_VERIFIED" for item in certificates)
        )
        self.assertTrue(all(item["cases_passed"] == 3 for item in certificates))
        self.assertTrue(all(item["confidence"] == 1.0 for item in certificates))


if __name__ == "__main__":
    unittest.main()
