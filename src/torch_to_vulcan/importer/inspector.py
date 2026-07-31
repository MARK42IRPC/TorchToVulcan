"""Inspect a direct ONNX model or models stored in a ZIP archive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, LargeZipFile, ZipFile, ZipInfo

import onnx
from google.protobuf.message import DecodeError
from onnx import AttributeProto, GraphProto, ModelProto

from .report import (
    GraphReport,
    InspectionReport,
    ModelError,
    ModelReport,
    OperatorReport,
    OpsetReport,
)


@dataclass(frozen=True, slots=True)
class InspectionLimits:
    max_model_count: int = 512
    max_model_bytes: int = 256 * 1024 * 1024
    max_total_model_bytes: int = 1024 * 1024 * 1024


class InspectionError(ValueError):
    """Raised when an input source cannot be inspected safely."""


def inspect_path(
    source_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Inspect a direct ``.onnx`` model or a ``.zip`` containing ONNX models."""

    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return inspect_onnx(path, limits=limits)
    if suffix == ".zip":
        return inspect_archive(path, limits=limits)
    raise InspectionError(
        f"unsupported input format {path.suffix or '(none)'}; expected .onnx or .zip"
    )


def inspect_onnx(
    model_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Inspect one ONNX model without resolving external tensor data."""

    path = Path(model_path)
    report = InspectionReport(source=str(path), source_type="onnx")
    try:
        size = path.stat().st_size
        if size > limits.max_model_bytes:
            report.errors.append(
                ModelError(path.name, f"model is {size} bytes; limit is {limits.max_model_bytes}")
            )
            return report
        model = onnx.load_model_from_string(path.read_bytes())
        report.models.append(_inspect_model(path.name, model))
    except FileNotFoundError as error:
        raise InspectionError(f"model does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"model cannot be read: {path}") from error
    except (DecodeError, OSError, ValueError) as error:
        report.errors.append(ModelError(path.name, str(error) or type(error).__name__))
    return report


def inspect_archive(
    archive_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Parse every ``.onnx`` entry in a ZIP and return an inspection report.

    Entries are read directly from the archive. Tensor external-data references
    are intentionally not resolved because operator discovery only needs the
    protobuf graph structure.
    """

    path = Path(archive_path)
    report = InspectionReport(source=str(path), source_type="zip")

    try:
        with ZipFile(path, "r", allowZip64=True) as archive:
            entries = _onnx_entries(archive, limits)
            for entry in entries:
                if entry.flag_bits & 0x1:
                    report.errors.append(ModelError(entry.filename, "encrypted ZIP entry"))
                    continue
                if entry.file_size > limits.max_model_bytes:
                    report.errors.append(
                        ModelError(
                            entry.filename,
                            f"model is {entry.file_size} bytes; limit is {limits.max_model_bytes}",
                        )
                    )
                    continue

                try:
                    model = onnx.load_model_from_string(archive.read(entry))
                    report.models.append(_inspect_model(entry.filename, model))
                except (BadZipFile, DecodeError, OSError, ValueError) as error:
                    report.errors.append(ModelError(entry.filename, str(error) or type(error).__name__))
    except FileNotFoundError as error:
        raise InspectionError(f"archive does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"archive cannot be read: {path}") from error
    except (BadZipFile, LargeZipFile) as error:
        raise InspectionError(f"invalid ZIP archive: {path}: {error}") from error

    return report


def _onnx_entries(archive: ZipFile, limits: InspectionLimits) -> list[ZipInfo]:
    entries = sorted(
        (
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and entry.filename.lower().endswith(".onnx")
        ),
        key=lambda entry: entry.filename.casefold(),
    )
    if len(entries) > limits.max_model_count:
        raise InspectionError(
            f"archive contains {len(entries)} ONNX models; limit is {limits.max_model_count}"
        )

    total_bytes = sum(entry.file_size for entry in entries)
    if total_bytes > limits.max_total_model_bytes:
        raise InspectionError(
            f"ONNX entries total {total_bytes} bytes; limit is {limits.max_total_model_bytes}"
        )
    return entries


def _inspect_model(model_path: str, model: ModelProto) -> ModelReport:
    root_name = model.graph.name or "main"
    graphs: list[GraphReport] = []
    _inspect_graph(model.graph, root_name, graphs)
    opsets = tuple(
        OpsetReport(domain=opset.domain, version=opset.version)
        for opset in sorted(model.opset_import, key=lambda item: item.domain)
    )
    return ModelReport(
        path=model_path,
        graph_name=root_name,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        opsets=opsets,
        graphs=tuple(graphs),
    )


def _inspect_graph(graph: GraphProto, path: str, reports: list[GraphReport]) -> None:
    operators = tuple(
        OperatorReport(
            graph_path=path,
            index=index,
            name=node.name,
            op_type=node.op_type,
            domain=node.domain,
            inputs=tuple(node.input),
            outputs=tuple(node.output),
        )
        for index, node in enumerate(graph.node)
    )
    reports.append(GraphReport(path=path, name=graph.name, operators=operators))

    for node_index, node in enumerate(graph.node):
        node_segment = node.name or f"{node.op_type}[{node_index}]"
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                child_path = f"{path}/{node_segment}.{attribute.name}"
                _inspect_graph(attribute.g, child_path, reports)
            elif attribute.type == AttributeProto.GRAPHS:
                for graph_index, child_graph in enumerate(attribute.graphs):
                    child_path = f"{path}/{node_segment}.{attribute.name}[{graph_index}]"
                    _inspect_graph(child_graph, child_path, reports)
