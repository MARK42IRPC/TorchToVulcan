"""Inspect ONNX files and common compressed containers."""

from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from zipfile import BadZipFile, LargeZipFile, ZipFile, ZipInfo

import onnx
import py7zr
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


_ARCHIVE_FORMATS = {
    ".zip": "zip",
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tgz": "tar",
    ".tar.bz2": "tar",
    ".tbz2": "tar",
    ".tar.xz": "tar",
    ".txz": "tar",
    ".7z": "7z",
}
_COMPRESSED_MODEL_FORMATS = {
    ".onnx.gz": "gzip",
    ".onnx.bz2": "bzip2",
    ".onnx.xz": "xz",
}


def supported_input_suffixes() -> tuple[str, ...]:
    """Return suffixes accepted by the file picker and HTTP adapter."""

    return (".onnx", *_COMPRESSED_MODEL_FORMATS, *_ARCHIVE_FORMATS)


def source_format(source_name: str | Path) -> str | None:
    """Return the importer format for a filename, or ``None`` if unsupported."""

    name = Path(source_name).name.lower()
    if name.endswith(".onnx"):
        return "onnx"
    for suffix, format_name in (*_COMPRESSED_MODEL_FORMATS.items(), *_ARCHIVE_FORMATS.items()):
        if name.endswith(suffix):
            return format_name
    return None


def inspect_path(
    source_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Inspect an ONNX file or a supported compressed container."""

    path = Path(source_path)
    format_name = source_format(path)
    if format_name == "onnx":
        return inspect_onnx(path, limits=limits)
    if format_name == "zip":
        return inspect_archive(path, limits=limits)
    if format_name == "tar":
        return inspect_tar(path, limits=limits)
    if format_name == "7z":
        return inspect_7z(path, limits=limits)
    if format_name in {"gzip", "bzip2", "xz"}:
        return inspect_compressed_onnx(path, format_name, limits=limits)
    supported = ", ".join(supported_input_suffixes())
    raise InspectionError(
        f"unsupported input format {path.suffix or '(none)'}; expected {supported}"
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
        _inspect_model_bytes(report, path.name, path.read_bytes())
    except FileNotFoundError as error:
        raise InspectionError(f"model does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"model cannot be read: {path}") from error
    except OSError as error:
        report.errors.append(ModelError(path.name, str(error) or type(error).__name__))
    return report


def inspect_compressed_onnx(
    model_path: str | Path,
    format_name: str,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Inspect one gzip, bzip2, or xz-compressed ONNX file."""

    path = Path(model_path)
    report = InspectionReport(source=str(path), source_type=format_name)
    try:
        compressed = path.read_bytes()
        if format_name == "gzip":
            stream = gzip.GzipFile(fileobj=BytesIO(compressed))
        elif format_name == "bzip2":
            stream = bz2.BZ2File(BytesIO(compressed))
        else:
            stream = lzma.LZMAFile(BytesIO(compressed))
        with stream:
            data = stream.read(limits.max_model_bytes + 1)
        _inspect_model_bytes(report, path.name, data, limits=limits)
    except FileNotFoundError as error:
        raise InspectionError(f"model does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"model cannot be read: {path}") from error
    except (OSError, EOFError, lzma.LZMAError, ValueError) as error:
        report.errors.append(ModelError(path.name, str(error) or type(error).__name__))
    return report


def inspect_archive(
    archive_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Parse every ONNX entry in a ZIP archive without extracting files."""

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
                _inspect_model_bytes(report, entry.filename, archive.read(entry), limits=limits)
    except FileNotFoundError as error:
        raise InspectionError(f"archive does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"archive cannot be read: {path}") from error
    except (BadZipFile, LargeZipFile) as error:
        raise InspectionError(f"invalid ZIP archive: {path}: {error}") from error
    return report


def inspect_tar(
    archive_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Parse ONNX entries in a TAR-family archive (including gzip/bzip2/xz)."""

    path = Path(archive_path)
    report = InspectionReport(source=str(path), source_type="tar")
    try:
        with tarfile.open(path, mode="r:*") as archive:
            entries = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.lower().endswith(".onnx")
                ),
                key=lambda member: member.name.casefold(),
            )
            _validate_archive_sizes(
                ((member.name, member.size) for member in entries), limits=limits
            )
            for entry in entries:
                if entry.size > limits.max_model_bytes:
                    report.errors.append(
                        ModelError(
                            entry.name,
                            f"model is {entry.size} bytes; limit is {limits.max_model_bytes}",
                        )
                    )
                    continue
                stream = archive.extractfile(entry)
                if stream is None:
                    report.errors.append(ModelError(entry.name, "TAR entry could not be read"))
                    continue
                _inspect_model_bytes(
                    report,
                    entry.name,
                    stream.read(limits.max_model_bytes + 1),
                    limits=limits,
                )
    except FileNotFoundError as error:
        raise InspectionError(f"archive does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"archive cannot be read: {path}") from error
    except (tarfile.ReadError, EOFError, OSError) as error:
        raise InspectionError(f"invalid TAR archive: {path}: {error}") from error
    return report


def inspect_7z(
    archive_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> InspectionReport:
    """Parse ONNX entries in a 7z archive using the pure-Python py7zr adapter."""

    path = Path(archive_path)
    report = InspectionReport(source=str(path), source_type="7z")
    try:
        with py7zr.SevenZipFile(path, mode="r") as archive:
            entries = sorted(
                (
                    info
                    for info in archive.list()
                    if not info.is_directory and info.filename.lower().endswith(".onnx")
                ),
                key=lambda info: info.filename.casefold(),
            )
            _validate_archive_sizes(
                ((info.filename, int(info.uncompressed or 0)) for info in entries), limits=limits
            )
            readable = [
                info for info in entries if int(info.uncompressed or 0) <= limits.max_model_bytes
            ]
            for info in entries:
                size = int(info.uncompressed or 0)
                if size > limits.max_model_bytes:
                    report.errors.append(
                        ModelError(
                            info.filename,
                            f"model is {size} bytes; limit is {limits.max_model_bytes}",
                        )
                    )
            payloads = archive.read(targets=[info.filename for info in readable]) or {}
            for info in readable:
                payload = payloads.get(info.filename)
                if payload is None:
                    report.errors.append(ModelError(info.filename, "7z entry could not be read"))
                    continue
                _inspect_model_bytes(report, info.filename, payload.read(), limits=limits)
    except FileNotFoundError as error:
        raise InspectionError(f"archive does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"archive cannot be read: {path}") from error
    except (py7zr.exceptions.Bad7zFile, EOFError, OSError, ValueError) as error:
        raise InspectionError(f"invalid 7z archive: {path}: {error}") from error
    return report


def _inspect_model_bytes(
    report: InspectionReport,
    model_path: str,
    data: bytes,
    *,
    limits: InspectionLimits = InspectionLimits(),
) -> None:
    if len(data) > limits.max_model_bytes:
        report.errors.append(
            ModelError(model_path, f"model exceeds {limits.max_model_bytes} byte limit")
        )
        return
    try:
        model = onnx.load_model_from_string(data)
        report.models.append(_inspect_model(model_path, model))
    except (DecodeError, OSError, ValueError) as error:
        report.errors.append(ModelError(model_path, str(error) or type(error).__name__))


def _validate_archive_sizes(
    entries: Iterable[tuple[str, int]],
    *,
    limits: InspectionLimits,
) -> None:
    values = list(entries)
    if len(values) > limits.max_model_count:
        raise InspectionError(
            f"archive contains {len(values)} ONNX models; limit is {limits.max_model_count}"
        )
    total_bytes = sum(size for _, size in values)
    if total_bytes > limits.max_total_model_bytes:
        raise InspectionError(
            f"ONNX entries total {total_bytes} bytes; limit is {limits.max_total_model_bytes}"
        )


def _onnx_entries(archive: ZipFile, limits: InspectionLimits) -> list[ZipInfo]:
    entries = sorted(
        (
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and entry.filename.lower().endswith(".onnx")
        ),
        key=lambda entry: entry.filename.casefold(),
    )
    _validate_archive_sizes(
        ((entry.filename, entry.file_size) for entry in entries), limits=limits
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
    reports.append(
        GraphReport(
            path=path,
            name=graph.name,
            inputs=tuple(value.name for value in graph.input),
            outputs=tuple(value.name for value in graph.output),
            operators=operators,
        )
    )

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
