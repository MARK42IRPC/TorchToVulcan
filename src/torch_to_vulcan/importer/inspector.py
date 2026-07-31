"""Inspect ONNX files and common compressed containers."""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable
from zipfile import BadZipFile, LargeZipFile, ZipFile, ZipInfo

import onnx
import psutil
import py7zr
import rarfile
from google.protobuf.message import DecodeError
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto, TypeProto, ValueInfoProto

from .report import (
    GraphReport,
    InspectionReport,
    ModelError,
    ModelReport,
    OperatorReport,
    OpsetReport,
    TensorValueReport,
)

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class InspectionLimits:
    max_model_count: int = 512
    max_model_bytes: int | None = None
    max_total_model_bytes: int | None = None
    memory_warning_ratio: float = 0.6


class InspectionError(ValueError):
    """Raised when an input source cannot be inspected safely."""


class MemoryConfirmationRequired(InspectionError):
    """Raised before parsing a model that is large relative to free memory."""

    def __init__(
        self,
        model_path: str,
        estimated_bytes: int,
        available_bytes: int,
        warning_ratio: float,
    ) -> None:
        self.model_path = model_path
        self.estimated_bytes = estimated_bytes
        self.available_bytes = available_bytes
        self.warning_ratio = warning_ratio
        super().__init__(
            f"{model_path} requires about {estimated_bytes} bytes while "
            f"{available_bytes} bytes are available"
        )


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
    ".rar": "rar",
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
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
) -> InspectionReport:
    """Inspect an ONNX file or a supported compressed container."""

    path = Path(source_path)
    _notify_progress(progress, "scanning", path.name, 0, 0, 0.0)
    format_name = source_format(path)
    if format_name == "onnx":
        return inspect_onnx(
            path, limits=limits, confirm_large_model=confirm_large_model, progress=progress
        )
    if format_name == "zip":
        return inspect_archive(
            path, limits=limits, confirm_large_model=confirm_large_model, progress=progress
        )
    if format_name == "tar":
        return inspect_tar(
            path, limits=limits, confirm_large_model=confirm_large_model, progress=progress
        )
    if format_name == "7z":
        return inspect_7z(
            path, limits=limits, confirm_large_model=confirm_large_model, progress=progress
        )
    if format_name == "rar":
        return inspect_rar(
            path, limits=limits, confirm_large_model=confirm_large_model, progress=progress
        )
    if format_name in {"gzip", "bzip2", "xz"}:
        return inspect_compressed_onnx(
            path,
            format_name,
            limits=limits,
            confirm_large_model=confirm_large_model,
            progress=progress,
        )
    supported = ", ".join(supported_input_suffixes())
    raise InspectionError(
        f"unsupported input format {path.suffix or '(none)'}; expected {supported}"
    )


def inspect_onnx(
    model_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
) -> InspectionReport:
    """Inspect one ONNX model without resolving external tensor data."""

    path = Path(model_path)
    report = InspectionReport(source=str(path), source_type="onnx")
    try:
        size = path.stat().st_size
        _notify_progress(progress, "discovered", path.name, 0, 1, 0.02)
        _require_memory_confirmation(path.name, size, limits, confirm_large_model)
        if limits.max_model_bytes is not None and size > limits.max_model_bytes:
            report.errors.append(
                ModelError(path.name, f"model is {size} bytes; limit is {limits.max_model_bytes}")
            )
            return report
        _notify_model_progress(progress, "reading", path.name, 0, 1)
        data = path.read_bytes()
        _notify_model_progress(progress, "parsing", path.name, 0, 1)
        _inspect_model_bytes(
            report,
            path.name,
            data,
            limits=limits,
            confirm_large_model=confirm_large_model,
        )
        _notify_model_progress(progress, "completed", path.name, 0, 1)
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
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
) -> InspectionReport:
    """Inspect one gzip, bzip2, or xz-compressed ONNX file."""

    path = Path(model_path)
    report = InspectionReport(source=str(path), source_type=format_name)
    try:
        compressed = path.read_bytes()
        _notify_progress(progress, "discovered", path.name, 0, 1, 0.02)
        _notify_model_progress(progress, "extracting", path.name, 0, 1)
        if format_name == "gzip":
            stream = gzip.GzipFile(fileobj=BytesIO(compressed))
        elif format_name == "bzip2":
            stream = bz2.BZ2File(BytesIO(compressed))
        else:
            stream = lzma.LZMAFile(BytesIO(compressed))
        with stream:
            data = stream.read(_read_size(limits))
        _require_memory_confirmation(path.name, len(data), limits, confirm_large_model)
        _notify_model_progress(progress, "parsing", path.name, 0, 1)
        _inspect_model_bytes(
            report,
            path.name,
            data,
            limits=limits,
            confirm_large_model=confirm_large_model,
        )
        _notify_model_progress(progress, "completed", path.name, 0, 1)
    except FileNotFoundError as error:
        raise InspectionError(f"model does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"model cannot be read: {path}") from error
    except MemoryConfirmationRequired:
        raise
    except (OSError, EOFError, lzma.LZMAError, ValueError) as error:
        report.errors.append(ModelError(path.name, str(error) or type(error).__name__))
    return report


def inspect_archive(
    archive_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
) -> InspectionReport:
    """Parse every ONNX entry in a ZIP archive without extracting files."""

    path = Path(archive_path)
    report = InspectionReport(source=str(path), source_type="zip")
    try:
        with ZipFile(path, "r", allowZip64=True) as archive:
            entries = _onnx_entries(archive, limits)
            _notify_progress(progress, "discovered", path.name, 0, len(entries), 0.02)
            if entries:
                largest = max(entries, key=lambda entry: entry.file_size)
                _require_memory_confirmation(
                    largest.filename, largest.file_size, limits, confirm_large_model
                )
            for index, entry in enumerate(entries):
                if entry.flag_bits & 0x1:
                    report.errors.append(ModelError(entry.filename, "encrypted ZIP entry"))
                    continue
                if limits.max_model_bytes is not None and entry.file_size > limits.max_model_bytes:
                    report.errors.append(
                        ModelError(
                            entry.filename,
                            f"model is {entry.file_size} bytes; limit is {limits.max_model_bytes}",
                        )
                    )
                    continue
                _notify_model_progress(
                    progress, "extracting", entry.filename, index, len(entries)
                )
                data = archive.read(entry)
                _notify_model_progress(progress, "parsing", entry.filename, index, len(entries))
                _inspect_model_bytes(
                    report,
                    entry.filename,
                    data,
                    limits=limits,
                    confirm_large_model=confirm_large_model,
                )
                _notify_model_progress(
                    progress, "completed", entry.filename, index, len(entries)
                )
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
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
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
            _notify_progress(progress, "discovered", path.name, 0, len(entries), 0.02)
            if entries:
                largest = max(entries, key=lambda entry: entry.size)
                _require_memory_confirmation(
                    largest.name, largest.size, limits, confirm_large_model
                )
            for index, entry in enumerate(entries):
                if limits.max_model_bytes is not None and entry.size > limits.max_model_bytes:
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
                _notify_model_progress(
                    progress, "extracting", entry.name, index, len(entries)
                )
                data = stream.read(_read_size(limits))
                _notify_model_progress(progress, "parsing", entry.name, index, len(entries))
                _inspect_model_bytes(
                    report,
                    entry.name,
                    data,
                    limits=limits,
                    confirm_large_model=confirm_large_model,
                )
                _notify_model_progress(progress, "completed", entry.name, index, len(entries))
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
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
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
            _notify_progress(progress, "discovered", path.name, 0, len(entries), 0.02)
            total_size = sum(int(info.uncompressed or 0) for info in entries)
            if entries:
                _require_memory_confirmation(
                    "7z ONNX payloads", total_size, limits, confirm_large_model
                )
            readable = [
                info
                for info in entries
                if limits.max_model_bytes is None
                or int(info.uncompressed or 0) <= limits.max_model_bytes
            ]
            for info in entries:
                size = int(info.uncompressed or 0)
                if limits.max_model_bytes is not None and size > limits.max_model_bytes:
                    report.errors.append(
                        ModelError(
                            info.filename,
                            f"model is {size} bytes; limit is {limits.max_model_bytes}",
                        )
                    )
            if readable:
                _notify_progress(progress, "extracting", path.name, 0, len(readable), 0.05)
            payloads = archive.read(targets=[info.filename for info in readable]) or {}
            for index, info in enumerate(readable):
                payload = payloads.get(info.filename)
                if payload is None:
                    report.errors.append(ModelError(info.filename, "7z entry could not be read"))
                    continue
                _notify_model_progress(
                    progress, "parsing", info.filename, index, len(readable)
                )
                _inspect_model_bytes(
                    report,
                    info.filename,
                    payload.read(),
                    limits=limits,
                    confirm_large_model=confirm_large_model,
                )
                _notify_model_progress(
                    progress, "completed", info.filename, index, len(readable)
                )
    except FileNotFoundError as error:
        raise InspectionError(f"archive does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"archive cannot be read: {path}") from error
    except MemoryConfirmationRequired:
        raise
    except (py7zr.exceptions.Bad7zFile, EOFError, OSError, ValueError) as error:
        raise InspectionError(f"invalid 7z archive: {path}: {error}") from error
    return report


def inspect_rar(
    archive_path: str | Path,
    *,
    limits: InspectionLimits = InspectionLimits(),
    confirm_large_model: bool = False,
    progress: ProgressCallback | None = None,
) -> InspectionReport:
    """Parse ONNX entries in a RAR archive with the bundled decoder."""

    path = Path(archive_path)
    report = InspectionReport(source=str(path), source_type="rar")
    _configure_rar_decoder()
    try:
        with rarfile.RarFile(path, mode="r") as archive:
            entries = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.isdir() and info.filename.lower().endswith(".onnx")
                ),
                key=lambda info: info.filename.casefold(),
            )
            _validate_archive_sizes(
                ((info.filename, info.file_size) for info in entries), limits=limits
            )
            _notify_progress(progress, "discovered", path.name, 0, len(entries), 0.02)
            if entries:
                largest = max(entries, key=lambda entry: entry.file_size)
                _require_memory_confirmation(
                    largest.filename, largest.file_size, limits, confirm_large_model
                )
            readable = []
            for info in entries:
                if info.needs_password():
                    report.errors.append(ModelError(info.filename, "encrypted RAR entry"))
                elif limits.max_model_bytes is not None and info.file_size > limits.max_model_bytes:
                    report.errors.append(
                        ModelError(
                            info.filename,
                            f"model is {info.file_size} bytes; limit is {limits.max_model_bytes}",
                        )
                    )
                else:
                    readable.append(info)

            basenames = [Path(info.filename.replace("\\", "/")).name for info in readable]
            if len(basenames) != len(set(name.casefold() for name in basenames)):
                raise InspectionError(
                    "RAR contains ONNX entries with duplicate basenames; flat safe extraction "
                    "cannot distinguish them"
                )

            with TemporaryDirectory(prefix="torch-to-vulcan-rar-") as extract_directory:
                if readable:
                    _notify_progress(
                        progress,
                        "extracting",
                        f"{len(readable)} ONNX models",
                        0,
                        len(readable),
                        0.05,
                    )
                    _extract_rar_models(path, Path(extract_directory))
                for index, (info, basename) in enumerate(zip(readable, basenames, strict=True)):
                    extracted_path = Path(extract_directory) / basename
                    if not extracted_path.is_file():
                        report.errors.append(
                            ModelError(info.filename, "RAR entry could not be extracted")
                        )
                        continue
                    _notify_model_progress(
                        progress, "parsing", info.filename, index, len(readable)
                    )
                    _inspect_model_bytes(
                        report,
                        info.filename,
                        extracted_path.read_bytes(),
                        limits=limits,
                        confirm_large_model=confirm_large_model,
                    )
                    _notify_model_progress(
                        progress, "completed", info.filename, index, len(readable)
                    )
    except FileNotFoundError as error:
        raise InspectionError(f"archive does not exist: {path}") from error
    except PermissionError as error:
        raise InspectionError(f"archive cannot be read: {path}") from error
    except rarfile.RarCannotExec as error:
        raise InspectionError(
            "RAR decoder is unavailable; reinstall dependencies or set TTV_UNRAR"
        ) from error
    except MemoryConfirmationRequired:
        raise
    except InspectionError:
        raise
    except (rarfile.Error, EOFError, OSError, ValueError) as error:
        raise InspectionError(f"invalid RAR archive: {path}: {error}") from error
    return report


def _configure_rar_decoder() -> None:
    """Select the bundled UnRAR first, then common system installations."""

    executable_name = "UnRAR.exe" if os.name == "nt" else "unrar"
    bundled = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "unrar"
        / ("windows-x86_64" if os.name == "nt" else "unix")
        / executable_name
    )
    configured = os.environ.get("TTV_UNRAR")
    candidates = [
        Path(configured).expanduser() if configured else None,
        bundled,
        Path(r"C:\Program Files\WinRAR\UnRAR.exe") if os.name == "nt" else None,
        Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe") if os.name == "nt" else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            rarfile.UNRAR_TOOL = str(candidate)
            return
    system_decoder = shutil.which("unrar") or shutil.which("UnRAR.exe")
    if system_decoder:
        rarfile.UNRAR_TOOL = system_decoder


def _extract_rar_models(archive_path: Path, destination: Path) -> None:
    command = [
        rarfile.UNRAR_TOOL,
        "e",
        "-idq",
        "-o+",
        "-p-",
        str(archive_path),
        "*.onnx",
        f"{destination}{os.sep}",
    ]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    result: subprocess.CompletedProcess[bytes] | None = None
    attempts = 2 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                creationflags=creation_flags,
            )
        except FileNotFoundError as error:
            raise rarfile.RarCannotExec("UnRAR executable was not found") from error
        if _windows_exit_code(result.returncode) != 0xC000013A or attempt + 1 == attempts:
            break

    assert result is not None
    if result.returncode not in {0, 1}:
        message = result.stderr.decode(errors="replace").strip()
        if _windows_exit_code(result.returncode) == 0xC000013A:
            raise InspectionError(
                "UnRAR was interrupted by a Windows console control signal after retry"
            )
        raise InspectionError(
            f"UnRAR extraction failed with exit code {result.returncode}: {message}"
        )


def _windows_exit_code(return_code: int) -> int:
    return return_code & 0xFFFFFFFF


def _inspect_model_bytes(
    report: InspectionReport,
    model_path: str,
    data: bytes,
    *,
    limits: InspectionLimits = InspectionLimits(),
    confirm_large_model: bool = False,
) -> None:
    if limits.max_model_bytes is not None and len(data) > limits.max_model_bytes:
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
    if (
        limits.max_total_model_bytes is not None
        and total_bytes > limits.max_total_model_bytes
    ):
        raise InspectionError(
            f"ONNX entries total {total_bytes} bytes; limit is {limits.max_total_model_bytes}"
        )


def _read_size(limits: InspectionLimits) -> int:
    return -1 if limits.max_model_bytes is None else limits.max_model_bytes + 1


def _require_memory_confirmation(
    model_path: str,
    estimated_bytes: int,
    limits: InspectionLimits,
    confirmed: bool,
) -> None:
    if confirmed or estimated_bytes <= 0:
        return
    available_bytes = int(psutil.virtual_memory().available)
    if estimated_bytes > available_bytes * limits.memory_warning_ratio:
        raise MemoryConfirmationRequired(
            model_path,
            estimated_bytes,
            available_bytes,
            limits.memory_warning_ratio,
        )


def _notify_model_progress(
    callback: ProgressCallback | None,
    phase: str,
    model_path: str,
    index: int,
    total: int,
) -> None:
    phase_fraction = {
        "reading": 0.1,
        "extracting": 0.1,
        "parsing": 0.75,
        "completed": 1.0,
    }[phase]
    completed = index + 1 if phase == "completed" else index
    fraction = 1.0 if total == 0 else (index + phase_fraction) / total
    _notify_progress(
        callback,
        phase,
        model_path,
        completed,
        total,
        min(1.0, 0.02 + 0.98 * fraction),
        current=index + 1,
    )


def _notify_progress(
    callback: ProgressCallback | None,
    phase: str,
    model_path: str,
    completed: int,
    total: int,
    fraction: float,
    *,
    current: int = 0,
) -> None:
    if callback is None:
        return
    callback(
        {
            "phase": phase,
            "model_path": model_path,
            "completed": completed,
            "current": current,
            "total": total,
            "percent": round(fraction * 100, 1),
        }
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
            values=_inspect_tensor_values(graph),
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


def _inspect_tensor_values(graph: GraphProto) -> tuple[TensorValueReport, ...]:
    values: dict[str, TensorValueReport] = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.name:
            values[value.name] = _value_info_report(value)

    for initializer in graph.initializer:
        if initializer.name and initializer.name not in values:
            values[initializer.name] = TensorValueReport(
                name=initializer.name,
                data_type=_tensor_data_type_name(initializer.data_type),
                shape=tuple(str(dimension) for dimension in initializer.dims),
            )

    for initializer in graph.sparse_initializer:
        if initializer.values.name and initializer.values.name not in values:
            values[initializer.values.name] = TensorValueReport(
                name=initializer.values.name,
                data_type=_tensor_data_type_name(initializer.values.data_type),
                shape=tuple(str(dimension) for dimension in initializer.dims),
            )
    return tuple(values.values())


def _value_info_report(value: ValueInfoProto) -> TensorValueReport:
    return TensorValueReport(
        name=value.name,
        data_type=_type_name(value.type),
        shape=_type_shape(value.type),
    )


def _type_name(value_type: TypeProto) -> str:
    if value_type.HasField("tensor_type"):
        return _tensor_data_type_name(value_type.tensor_type.elem_type)
    if value_type.HasField("sparse_tensor_type"):
        return f"SPARSE<{_tensor_data_type_name(value_type.sparse_tensor_type.elem_type)}>"
    if value_type.HasField("sequence_type"):
        return f"SEQUENCE<{_type_name(value_type.sequence_type.elem_type)}>"
    if value_type.HasField("optional_type"):
        return f"OPTIONAL<{_type_name(value_type.optional_type.elem_type)}>"
    if value_type.HasField("map_type"):
        key_type = _tensor_data_type_name(value_type.map_type.key_type)
        return f"MAP<{key_type}, {_type_name(value_type.map_type.value_type)}>"
    return "UNKNOWN"


def _type_shape(value_type: TypeProto) -> tuple[str, ...]:
    if value_type.HasField("tensor_type"):
        return _shape_dimensions(value_type.tensor_type.shape.dim)
    if value_type.HasField("sparse_tensor_type"):
        return _shape_dimensions(value_type.sparse_tensor_type.shape.dim)
    if value_type.HasField("sequence_type"):
        return _type_shape(value_type.sequence_type.elem_type)
    if value_type.HasField("optional_type"):
        return _type_shape(value_type.optional_type.elem_type)
    return ()


def _shape_dimensions(dimensions: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for dimension in dimensions:
        if dimension.HasField("dim_value"):
            result.append(str(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return tuple(result)


def _tensor_data_type_name(data_type: int) -> str:
    if not data_type:
        return "UNKNOWN"
    try:
        return TensorProto.DataType.Name(data_type)
    except (KeyError, ValueError):
        return f"TYPE_{data_type}"
