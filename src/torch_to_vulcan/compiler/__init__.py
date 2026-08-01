"""Compiler intermediate representations, lowering registries, and packages."""

from .package import (
    ExecutablePackageBuilder,
    ExecutablePackageError,
    load_executable_manifest,
    validate_executable_package,
)
from .constants import (
    ConstantFoldResult,
    FoldDiagnostic,
    FoldedNode,
    fold_constant_subgraph,
    materialize_folded_constants,
    rewrite_model_with_folded_constants,
)
from .model import (
    CompilationDiagnostic,
    CompilationReport,
    StaticCompilationError,
    compile_static_model,
    compile_static_onnx,
)
from .contracts import (
    BackendCapabilities,
    ContractError,
    OperatorCapability,
    ShapeProfile,
)
from .onnx import (
    NormalizedAttributeTensor,
    NormalizedConstant,
    NormalizedGraph,
    NormalizedModel,
    NormalizedNode,
    NormalizedTensor,
    NormalizationError,
    normalize_onnx_model,
)

__all__ = [
    "ExecutablePackageBuilder",
    "ExecutablePackageError",
    "load_executable_manifest",
    "validate_executable_package",
    "ConstantFoldResult",
    "FoldDiagnostic",
    "FoldedNode",
    "fold_constant_subgraph",
    "materialize_folded_constants",
    "rewrite_model_with_folded_constants",
    "CompilationDiagnostic",
    "CompilationReport",
    "StaticCompilationError",
    "compile_static_model",
    "compile_static_onnx",
    "BackendCapabilities",
    "ContractError",
    "OperatorCapability",
    "ShapeProfile",
    "NormalizedAttributeTensor",
    "NormalizedConstant",
    "NormalizedGraph",
    "NormalizedModel",
    "NormalizedNode",
    "NormalizedTensor",
    "NormalizationError",
    "normalize_onnx_model",
]
