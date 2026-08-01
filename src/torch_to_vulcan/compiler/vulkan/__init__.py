"""Vulkan kernel planning and verification entry points."""

from .executor import VulkanExecutionError, VulkanExecutionResult, VulkanExecutor
from ..contracts import BackendCapabilities, OperatorCapability
from .ir import BufferBinding, DispatchPlan, DispatchStep, ShaderModule
from .kernels import KernelRegistry, default_kernel_registry
from .verify import (
    ToolchainCapabilities,
    VerificationRunner,
    VerificationTarget,
    VerificationTensor,
    detect_toolchain,
)

__all__ = [
    "BufferBinding",
    "BackendCapabilities",
    "DispatchPlan",
    "DispatchStep",
    "KernelRegistry",
    "OperatorCapability",
    "ShaderModule",
    "ToolchainCapabilities",
    "VulkanExecutionError",
    "VulkanExecutionResult",
    "VulkanExecutor",
    "PackageBenchmark",
    "PackageExecutionResult",
    "VulkanPackageRuntime",
    "VerificationRunner",
    "VerificationTarget",
    "VerificationTensor",
    "default_kernel_registry",
    "detect_toolchain",
]


def __getattr__(name: str):
    if name in {"PackageBenchmark", "PackageExecutionResult", "VulkanPackageRuntime"}:
        from .runtime import PackageBenchmark, PackageExecutionResult, VulkanPackageRuntime

        return {
            "PackageBenchmark": PackageBenchmark,
            "PackageExecutionResult": PackageExecutionResult,
            "VulkanPackageRuntime": VulkanPackageRuntime,
        }[name]
    raise AttributeError(name)
