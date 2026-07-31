"""ONNX file and archive inspection APIs."""

from .inspector import (
    InspectionError,
    InspectionLimits,
    inspect_7z,
    inspect_archive,
    inspect_onnx,
    inspect_path,
    inspect_tar,
    source_format,
    supported_input_suffixes,
)
from .report import (
    GraphReport,
    InspectionReport,
    ModelError,
    ModelReport,
    OperatorCount,
    OperatorReport,
    OpsetReport,
)

__all__ = [
    "GraphReport",
    "InspectionError",
    "InspectionLimits",
    "InspectionReport",
    "ModelError",
    "ModelReport",
    "OperatorCount",
    "OperatorReport",
    "OpsetReport",
    "inspect_7z",
    "inspect_archive",
    "inspect_onnx",
    "inspect_path",
    "inspect_tar",
    "source_format",
    "supported_input_suffixes",
]
