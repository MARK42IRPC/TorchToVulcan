"""ONNX file and archive inspection APIs."""

from .inspector import InspectionError, InspectionLimits, inspect_archive, inspect_onnx, inspect_path
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
    "inspect_archive",
    "inspect_onnx",
    "inspect_path",
]
