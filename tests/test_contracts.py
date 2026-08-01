from __future__ import annotations

import unittest

from torch_to_vulcan.compiler import (
    BackendCapabilities,
    ContractError,
    OperatorCapability,
    ShapeProfile,
)
from torch_to_vulcan.compiler.vulkan.kernels import (
    KernelCandidate,
    KernelContext,
    KernelRegistry,
    UnsupportedKernel,
    default_kernel_registry,
)
from torch_to_vulcan.compiler.vulkan.ir import DispatchPlan


class CompilerContractTests(unittest.TestCase):
    def test_shape_profile_resolves_symbolic_dimensions_and_round_trips(self) -> None:
        profile = ShapeProfile.from_mapping(
            {"batch": 1, "sequence": 128}, name="batch1-sequence128"
        )

        self.assertEqual(profile.resolve_shape(("batch", 4, "sequence")), (1, 4, 128))
        self.assertEqual(ShapeProfile.from_dict(profile.to_dict()), profile)

    def test_shape_profile_rejects_unbound_or_invalid_dimensions(self) -> None:
        profile = ShapeProfile.from_mapping({"batch": 1})

        with self.assertRaises(ContractError):
            profile.resolve_shape(("sequence",))
        with self.assertRaises(ContractError):
            ShapeProfile.from_mapping({"batch": -1})

    def test_backend_capabilities_expose_registered_kernel_matrix(self) -> None:
        capabilities = BackendCapabilities.from_registry(default_kernel_registry())

        relu = capabilities.support_for(
            domain="",
            op_type="Relu",
            opset_version=18,
            data_types=("FLOAT", "FLOAT"),
        )
        self.assertIsNotNone(relu)
        self.assertEqual(relu.op_type, "Relu")
        self.assertIsNone(
            capabilities.support_for(
                domain="",
                op_type="Relu",
                opset_version=18,
                data_types=("FLOAT16", "FLOAT16"),
            )
        )

    def test_capability_rejects_layout_and_opset_outside_declared_range(self) -> None:
        capability = OperatorCapability(
            domain="",
            op_type="Test",
            min_opset=10,
            max_opset=12,
            data_types=frozenset({"FLOAT"}),
            layouts=frozenset({"contiguous"}),
        )

        self.assertTrue(
            capability.matches(
                domain="",
                op_type="Test",
                opset_version=11,
                data_types=("FLOAT",),
                layout="contiguous",
            )
        )
        self.assertFalse(
            capability.matches(
                domain="",
                op_type="Test",
                opset_version=13,
                data_types=("FLOAT",),
                layout="strided",
            )
        )

    def test_kernel_selection_enforces_explicit_capability_contract(self) -> None:
        registry = KernelRegistry()
        registry.register(
            KernelCandidate(
                "test.fp32",
                "",
                "Test",
                lambda _context: DispatchPlan("test.fp32", "Test", ()),
                capability=OperatorCapability(
                    domain="",
                    op_type="Test",
                    data_types=frozenset({"FLOAT"}),
                ),
            )
        )

        with self.assertRaises(UnsupportedKernel):
            registry.select(
                KernelContext(
                    "",
                    "Test",
                    18,
                    {},
                    (),
                    (),
                    layout="strided",
                )
            )


if __name__ == "__main__":
    unittest.main()
