"""Writer and integrity validator for TTV executable directory packages."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .vulkan.ir import DispatchStep

FORMAT_VERSION = "0.1"
CONSTANT_ALIGNMENT = 256
SPIRV_MAGIC = b"\x03\x02\x23\x07"
DATA_TYPE_BYTES = {
    "BOOL": 1,
    "BFLOAT16": 2,
    "FLOAT16": 2,
    "FLOAT": 4,
    "DOUBLE": 8,
    "INT8": 1,
    "UINT8": 1,
    "INT16": 2,
    "UINT16": 2,
    "INT32": 4,
    "UINT32": 4,
    "INT64": 8,
    "UINT64": 8,
}


class ExecutablePackageError(ValueError):
    """A package cannot be written or trusted as an executable."""


class ExecutablePackageBuilder:
    """Collect static programs and materialize them atomically.

    ``main`` and ``self.steps`` remain the compatibility surface for TTV 0.1
    callers.  Additional named programs, profiles, state tensors, and host
    loops are opt-in manifest records.
    """

    def __init__(
        self,
        model_name: str,
        *,
        vulkan_api_version: str = "1.0",
        required_features: Sequence[str] = (),
    ) -> None:
        if not model_name:
            raise ExecutablePackageError("model_name must not be empty")
        self.model_name = model_name
        self.vulkan_api_version = vulkan_api_version
        self.required_features = tuple(dict.fromkeys(required_features))
        self.tensors: dict[str, dict[str, object]] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.constant_data = bytearray()
        self.has_constants = False
        self.shaders: dict[str, tuple[bytes, str, str]] = {}
        self.pipelines: dict[str, dict[str, object]] = {}
        self.programs: dict[str, dict[str, object]] = {
            "main": {"id": "main", "kind": "linear", "steps": []}
        }
        self.steps: list[dict[str, object]] = []
        self.certificates: dict[str, dict[str, object]] = {}
        self.profiles: dict[str, dict[str, object]] = {}
        self.states: dict[str, dict[str, object]] = {}
        self.loops: dict[str, dict[str, object]] = {}

    def add_tensor(
        self,
        tensor_id: str,
        data_type: str,
        shape: Sequence[int],
        *,
        storage: str = "transient",
    ) -> None:
        if storage not in {"external", "transient"}:
            raise ExecutablePackageError(f"unsupported tensor storage {storage}")
        self._add_tensor_record(
            tensor_id,
            data_type,
            shape,
            {"kind": storage},
        )

    def add_profile(
        self,
        profile_id: str,
        dimensions: Mapping[str, int] | None = None,
    ) -> None:
        """Record the concrete symbolic-dimension bindings used at compile time."""
        if not profile_id or profile_id in self.profiles:
            raise ExecutablePackageError(f"duplicate or empty profile ID {profile_id!r}")
        normalized: dict[str, int] = {}
        for symbol, value in dict(dimensions or {}).items():
            if not isinstance(symbol, str) or not symbol:
                raise ExecutablePackageError("profile dimension names must not be empty")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutablePackageError(
                    f"profile dimension {symbol!r} must be a non-negative integer"
                )
            normalized[symbol] = value
        self.profiles[profile_id] = {
            "id": profile_id,
            "dimensions": normalized,
        }

    def add_program(
        self,
        program_id: str,
        *,
        kind: str = "subprogram",
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
    ) -> None:
        """Add a named callable program that shares package tensor storage."""
        if not program_id or program_id in self.programs:
            raise ExecutablePackageError(f"duplicate or empty program ID {program_id!r}")
        if kind not in {"linear", "subprogram"}:
            raise ExecutablePackageError(f"unsupported program kind {kind}")
        input_ids = list(inputs)
        output_ids = list(outputs)
        if len(set(input_ids)) != len(input_ids) or len(set(output_ids)) != len(output_ids):
            raise ExecutablePackageError(f"program {program_id} bindings must be unique")
        self.programs[program_id] = {
            "id": program_id,
            "kind": kind,
            "steps": [],
            "bindings": {"inputs": input_ids, "outputs": output_ids},
        }

    def add_subprogram(
        self,
        program_id: str,
        *,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
    ) -> None:
        self.add_program(program_id, inputs=inputs, outputs=outputs)

    def add_state_tensor(
        self,
        tensor_id: str,
        data_type: str,
        shape: Sequence[int],
        *,
        state_id: str | None = None,
        update_program: str | None = None,
        layout: str = "contiguous",
        update_region: Mapping[str, object] | None = None,
    ) -> None:
        """Add a zero-initialized tensor that survives repeated program calls."""
        resolved_state_id = state_id or tensor_id
        if not resolved_state_id or resolved_state_id in self.states:
            raise ExecutablePackageError(
                f"duplicate or empty state ID {resolved_state_id!r}"
            )
        if not layout:
            raise ExecutablePackageError("state tensor layout must not be empty")
        region = dict(update_region or {"kind": "whole_tensor"})
        if region.get("kind") not in {"whole_tensor", "prefix_append"}:
            raise ExecutablePackageError(
                f"unsupported state update region {region.get('kind')!r}"
            )
        self._add_tensor_record(
            tensor_id,
            data_type,
            shape,
            {
                "kind": "state",
                "state_id": resolved_state_id,
                "initialization": "zero",
            },
        )
        update: dict[str, object] = {
            "mode": "in_place",
            "boundary": "invocation",
            "region": region,
        }
        if update_program is not None:
            update["program"] = update_program
        state: dict[str, object] = {
            "id": resolved_state_id,
            "tensor_id": tensor_id,
            "lifetime": "session",
            "reset": "zero",
            "layout": layout,
            "update": update,
        }
        self.states[resolved_state_id] = state

    def add_host_loop(
        self,
        loop_id: str,
        program_id: str,
        *,
        max_iterations: int,
        stop_tensor: str,
        comparison: str = "nonzero",
    ) -> None:
        """Describe a host-synchronized loop around a callable program."""
        if not loop_id or loop_id in self.loops:
            raise ExecutablePackageError(f"duplicate or empty loop ID {loop_id!r}")
        if max_iterations <= 0:
            raise ExecutablePackageError("host loop max_iterations must be positive")
        if comparison != "nonzero":
            raise ExecutablePackageError(
                f"unsupported host loop comparison {comparison!r}"
            )
        program = self.programs.get(program_id)
        if program is None:
            raise ExecutablePackageError(f"unknown loop program {program_id}")
        if program_id == "main" or program["kind"] != "subprogram":
            raise ExecutablePackageError("host loop program must be a subprogram")
        if stop_tensor not in self.tensors:
            raise ExecutablePackageError(f"unknown loop termination tensor {stop_tensor}")
        bindings = program["bindings"]
        if stop_tensor not in bindings["outputs"]:  # type: ignore[index]
            raise ExecutablePackageError(
                f"loop termination tensor {stop_tensor} must be a program output"
            )
        self.loops[loop_id] = {
            "id": loop_id,
            "kind": "host",
            "program": program_id,
            "max_iterations": max_iterations,
            "termination": {
                "kind": "host_tensor",
                "tensor_id": stop_tensor,
                "comparison": comparison,
            },
            "synchronization": "after_each_iteration",
        }

    def add_constant(
        self,
        tensor_id: str,
        data_type: str,
        shape: Sequence[int],
        data: bytes | bytearray | memoryview,
        *,
        alignment: int = CONSTANT_ALIGNMENT,
    ) -> None:
        if alignment <= 0 or alignment & (alignment - 1):
            raise ExecutablePackageError("constant alignment must be a positive power of two")
        payload = bytes(data)
        self.has_constants = True
        dimensions = tuple(shape)
        if data_type in DATA_TYPE_BYTES:
            expected_length = math.prod(dimensions) * DATA_TYPE_BYTES[data_type]
            if len(payload) != expected_length:
                raise ExecutablePackageError(
                    f"constant {tensor_id} has {len(payload)} bytes, expected {expected_length}"
                )
        offset = _align(len(self.constant_data), alignment)
        self.constant_data.extend(b"\0" * (offset - len(self.constant_data)))
        self.constant_data.extend(payload)
        self._add_tensor_record(
            tensor_id,
            data_type,
            dimensions,
            {
                "kind": "constant",
                "blob_id": "constants",
                "offset": offset,
                "length": len(payload),
                "alignment": alignment,
            },
        )

    def add_view(
        self,
        tensor_id: str,
        data_type: str,
        shape: Sequence[int],
        source_tensor: str,
        strides: Sequence[int],
        *,
        byte_offset: int = 0,
    ) -> None:
        if source_tensor not in self.tensors:
            raise ExecutablePackageError(f"unknown view source tensor {source_tensor}")
        if byte_offset < 0 or any(value < 0 for value in strides):
            raise ExecutablePackageError("view offset and strides must be non-negative")
        if len(tuple(strides)) != len(tuple(shape)):
            raise ExecutablePackageError("view strides must match tensor rank")
        self._add_tensor_record(
            tensor_id,
            data_type,
            shape,
            {
                "kind": "view",
                "source_tensor": source_tensor,
                "byte_offset": byte_offset,
                "strides": list(strides),
            },
        )

    def bind_input(self, tensor_id: str) -> None:
        self._bind_tensor(self.inputs, tensor_id, "input")

    def bind_output(self, tensor_id: str) -> None:
        self._bind_tensor(self.outputs, tensor_id, "output")

    def add_dispatch(
        self,
        node_id: str,
        kernel_id: str,
        step: DispatchStep,
        spirv: bytes,
        resource_tensors: Sequence[str],
        *,
        certificate: Mapping[str, object] | None = None,
        program_id: str = "main",
    ) -> None:
        if program_id not in self.programs:
            raise ExecutablePackageError(f"unknown program {program_id}")
        program_steps = self.programs[program_id]["steps"]
        if not node_id or any(
            item["node_id"] == node_id
            for program in self.programs.values()
            for item in program["steps"]  # type: ignore[index]
        ):
            raise ExecutablePackageError(f"duplicate or empty dispatch node ID {node_id!r}")
        if len(resource_tensors) != len(step.bindings):
            raise ExecutablePackageError(
                f"dispatch {node_id} has {len(step.bindings)} bindings but "
                f"{len(resource_tensors)} tensor resources"
            )
        missing = [tensor_id for tensor_id in resource_tensors if tensor_id not in self.tensors]
        if missing:
            raise ExecutablePackageError(
                f"dispatch {node_id} references unknown tensors: {', '.join(missing)}"
            )
        if len(spirv) < 4 or len(spirv) % 4 or spirv[:4] != SPIRV_MAGIC:
            raise ExecutablePackageError("shader payload is not a SPIR-V module")

        shader_id = _sha256(spirv)
        self.shaders.setdefault(
            shader_id,
            (bytes(spirv), step.shader.source, step.shader.entry_point),
        )
        descriptors = [
            {
                "binding": binding.binding,
                "kind": "storage_buffer",
                "access": binding.access,
                "data_type": binding.data_type,
            }
            for binding in step.bindings
        ]
        push_layout = [
            {"name": name, "data_type": _push_constant_type(value)}
            for name, value in step.push_constants.items()
        ]
        pipeline_payload = {
            "kernel_id": kernel_id,
            "shader_id": shader_id,
            "descriptor_layout": descriptors,
            "push_constant_layout": push_layout,
        }
        pipeline_hash = _sha256(_canonical_json(pipeline_payload))
        pipeline_id = f"pipeline-{pipeline_hash}"
        self.pipelines.setdefault(pipeline_id, {"id": pipeline_id, **pipeline_payload})

        dispatch: dict[str, object] = {
            "kind": "dispatch",
            "node_id": node_id,
            "pipeline_id": pipeline_id,
            "workgroups": list(step.workgroups),
            "resources": [
                {"binding": binding.binding, "tensor_id": tensor_id}
                for binding, tensor_id in zip(
                    step.bindings,
                    resource_tensors,
                    strict=True,
                )
            ],
            "push_constants": dict(step.push_constants),
        }
        if certificate is not None:
            certificate_payload = deepcopy(dict(certificate))
            certificate_hash = _sha256(_canonical_json(certificate_payload))
            certificate_id = f"certificate-{certificate_hash}"
            certificate_payload["id"] = certificate_id
            self.certificates.setdefault(certificate_id, certificate_payload)
            dispatch["certificate_id"] = certificate_id
        program_steps.append(dispatch)  # type: ignore[union-attr]
        if program_id == "main":
            self.steps.append(dispatch)

    def write(
        self,
        directory: str | Path,
        *,
        include_debug_sources: bool = True,
        metadata: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        destination = Path(directory)
        if destination.exists():
            raise ExecutablePackageError(f"package destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            )
        )
        try:
            manifest = self._write_contents(
                temporary,
                include_debug_sources=include_debug_sources,
                metadata=metadata,
            )
            validate_executable_package(temporary)
            temporary.replace(destination)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _write_contents(
        self,
        root: Path,
        *,
        include_debug_sources: bool,
        metadata: Mapping[str, str] | None,
    ) -> dict[str, object]:
        blobs: list[dict[str, object]] = []
        if self.has_constants:
            constant_path = root / "constants" / "weights.bin"
            constant_path.parent.mkdir(parents=True, exist_ok=True)
            constant_bytes = bytes(self.constant_data)
            constant_path.write_bytes(constant_bytes)
            blobs.append(
                {
                    "id": "constants",
                    "file": "constants/weights.bin",
                    "length": len(constant_bytes),
                    "sha256": _sha256(constant_bytes),
                }
            )

        shader_records: list[dict[str, object]] = []
        for shader_id, (spirv, source, entry_point) in sorted(self.shaders.items()):
            relative_file = f"shaders/{shader_id}.spv"
            shader_path = root / relative_file
            shader_path.parent.mkdir(parents=True, exist_ok=True)
            shader_path.write_bytes(spirv)
            record: dict[str, object] = {
                "id": shader_id,
                "file": relative_file,
                "entry_point": entry_point,
                "sha256": shader_id,
            }
            if include_debug_sources:
                debug_file = f"debug/shaders/{shader_id}.comp"
                debug_path = root / debug_file
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(source, encoding="utf-8")
                record["debug_source"] = debug_file
            shader_records.append(record)

        certificate_file = root / "certificates" / "kernels.json"
        certificate_file.parent.mkdir(parents=True, exist_ok=True)
        certificate_bytes = _pretty_json(list(self.certificates.values()))
        certificate_file.write_bytes(certificate_bytes)
        manifest: dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "model_name": self.model_name,
            "target": {
                "vulkan_api_version": self.vulkan_api_version,
                "required_features": list(self.required_features),
            },
            "entry_program": "main",
            "bindings": {
                "inputs": list(self.inputs),
                "outputs": list(self.outputs),
            },
            "tensors": deepcopy(self.tensors),
            "blobs": blobs,
            "shaders": shader_records,
            "pipelines": list(self.pipelines.values()),
            "programs": [deepcopy(program) for program in self.programs.values()],
            "certificate_store": {
                "file": "certificates/kernels.json",
                "sha256": _sha256(certificate_bytes),
                "count": len(self.certificates),
            },
        }
        if metadata:
            manifest["metadata"] = dict(metadata)
        if self.profiles:
            manifest["profiles"] = deepcopy(list(self.profiles.values()))
        if self.states:
            manifest["states"] = deepcopy(list(self.states.values()))
        if self.loops:
            manifest["loops"] = deepcopy(list(self.loops.values()))
        (root / "manifest.json").write_bytes(_pretty_json(manifest))
        return manifest

    def _add_tensor_record(
        self,
        tensor_id: str,
        data_type: str,
        shape: Sequence[int],
        storage: Mapping[str, object],
    ) -> None:
        if not tensor_id or tensor_id in self.tensors:
            raise ExecutablePackageError(f"duplicate or empty tensor ID {tensor_id!r}")
        dimensions = tuple(shape)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in dimensions):
            raise ExecutablePackageError("package tensors require non-negative static dimensions")
        if not data_type:
            raise ExecutablePackageError("tensor data_type must not be empty")
        self.tensors[tensor_id] = {
            "data_type": data_type,
            "shape": list(dimensions),
            "shape_known": True,
            "storage": dict(storage),
        }

    def _bind_tensor(self, bindings: list[str], tensor_id: str, role: str) -> None:
        if tensor_id not in self.tensors:
            raise ExecutablePackageError(f"unknown {role} tensor {tensor_id}")
        if tensor_id in bindings:
            raise ExecutablePackageError(f"duplicate {role} tensor {tensor_id}")
        bindings.append(tensor_id)


def load_executable_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutablePackageError(f"cannot read package manifest: {error}") from error
    if not isinstance(value, dict):
        raise ExecutablePackageError("package manifest must be an object")
    return value


def validate_executable_package(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    manifest = load_executable_manifest(root)
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "model-package.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(manifest)
    except OSError as error:
        raise ExecutablePackageError(f"package schema is unavailable: {error}") from error
    except Exception as error:
        if type(error).__module__.startswith("jsonschema"):
            raise ExecutablePackageError(f"manifest schema validation failed: {error}") from error
        raise

    tensors = _mapping(manifest, "tensors")
    blobs = _unique_records(manifest, "blobs")
    shaders = _unique_records(manifest, "shaders")
    pipelines = _unique_records(manifest, "pipelines")
    programs = _unique_records(manifest, "programs")
    states = _optional_unique_records(manifest, "states")
    loops = _optional_unique_records(manifest, "loops")

    for tensor_id in (*manifest["bindings"]["inputs"], *manifest["bindings"]["outputs"]):
        _require_reference(tensors, tensor_id, "bound tensor")
    _require_reference(programs, manifest["entry_program"], "entry program")

    for program_id, program in programs.items():
        bindings = program.get("bindings")
        if program["kind"] == "subprogram" and bindings is None:
            raise ExecutablePackageError(f"subprogram {program_id} is missing bindings")
        if bindings is not None:
            for tensor_id in (*bindings["inputs"], *bindings["outputs"]):
                _require_reference(tensors, tensor_id, f"program {program_id} tensor")
    for state_id, state in states.items():
        tensor_id = state["tensor_id"]
        tensor = _require_reference(tensors, tensor_id, f"state {state_id} tensor")
        storage = tensor["storage"]
        if storage["kind"] != "state" or storage["state_id"] != state_id:
            raise ExecutablePackageError(
                f"state {state_id} does not match tensor {tensor_id} storage"
            )
        update_program = state["update"].get("program")
        if update_program is not None:
            program = _require_reference(
                programs, update_program, f"state {state_id} update program"
            )
            if program["kind"] != "subprogram":
                raise ExecutablePackageError(
                    f"state {state_id} update program must be a subprogram"
                )
            bindings = program["bindings"]
            if tensor_id not in bindings["inputs"] or tensor_id not in bindings["outputs"]:
                raise ExecutablePackageError(
                    f"state {state_id} update program must bind its tensor as input and output"
                )
        region = state["update"]["region"]
        if region["kind"] == "prefix_append":
            axis = region["axis"]
            if axis >= len(tensor["shape"]):
                raise ExecutablePackageError(
                    f"state {state_id} update axis {axis} exceeds tensor rank"
                )
            _require_reference(
                tensors,
                region["position_tensor"],
                f"state {state_id} position tensor",
            )

    for loop_id, loop in loops.items():
        program = _require_reference(programs, loop["program"], f"loop {loop_id} program")
        if program["kind"] != "subprogram":
            raise ExecutablePackageError(
                f"loop {loop_id} program must be a subprogram"
            )
        _require_reference(
            tensors,
            loop["termination"]["tensor_id"],
            f"loop {loop_id} termination tensor",
        )
        if loop["termination"]["tensor_id"] not in program.get("bindings", {}).get(
            "outputs", []
        ):
            raise ExecutablePackageError(
                f"loop {loop_id} termination tensor must be a program output"
            )

    blob_sizes: dict[str, int] = {}
    for blob_id, blob in blobs.items():
        payload = _verified_artifact(root, blob["file"], blob["sha256"])
        if len(payload) != blob["length"]:
            raise ExecutablePackageError(f"blob {blob_id} length does not match manifest")
        blob_sizes[blob_id] = len(payload)

    for tensor_id, tensor in tensors.items():
        storage = tensor["storage"]
        if storage["kind"] == "constant":
            blob_id = storage["blob_id"]
            _require_reference(blobs, blob_id, f"constant tensor {tensor_id} blob")
            if storage["offset"] % storage["alignment"]:
                raise ExecutablePackageError(f"constant tensor {tensor_id} is misaligned")
            if storage["offset"] + storage["length"] > blob_sizes[blob_id]:
                raise ExecutablePackageError(f"constant tensor {tensor_id} exceeds its blob")
        elif storage["kind"] == "view":
            _require_reference(tensors, storage["source_tensor"], f"view tensor {tensor_id}")
        elif storage["kind"] == "state":
            _require_reference(states, storage["state_id"], f"state tensor {tensor_id}")

    for shader_id, shader in shaders.items():
        payload = _verified_artifact(root, shader["file"], shader["sha256"])
        if shader_id != shader["sha256"]:
            raise ExecutablePackageError(f"shader {shader_id} is not content-addressed")
        if len(payload) < 4 or len(payload) % 4 or payload[:4] != SPIRV_MAGIC:
            raise ExecutablePackageError(f"shader {shader_id} is not valid SPIR-V data")
        if "debug_source" in shader:
            _artifact_path(root, shader["debug_source"], require_file=True)

    for pipeline_id, pipeline in pipelines.items():
        _require_reference(shaders, pipeline["shader_id"], f"pipeline {pipeline_id} shader")
        bindings = [item["binding"] for item in pipeline["descriptor_layout"]]
        if len(bindings) != len(set(bindings)):
            raise ExecutablePackageError(f"pipeline {pipeline_id} has duplicate bindings")

    certificate_store = manifest["certificate_store"]
    certificate_bytes = _verified_artifact(
        root,
        certificate_store["file"],
        certificate_store["sha256"],
    )
    try:
        certificate_values = json.loads(certificate_bytes)
    except json.JSONDecodeError as error:
        raise ExecutablePackageError("certificate store is not valid JSON") from error
    if not isinstance(certificate_values, list):
        raise ExecutablePackageError("certificate store must contain an array")
    if len(certificate_values) != certificate_store["count"]:
        raise ExecutablePackageError("certificate count does not match manifest")
    certificates = _unique_values(certificate_values, "certificate")

    seen_nodes: set[str] = set()
    for program_id, program in programs.items():
        bindings = program.get("bindings")
        if bindings is not None:
            resource_tensor_ids = {
                resource["tensor_id"]
                for step in program["steps"]
                for resource in step["resources"]
            }
            missing_outputs = set(bindings["outputs"]) - resource_tensor_ids
            if missing_outputs:
                raise ExecutablePackageError(
                    f"program {program_id} outputs are not bound by a dispatch: "
                    + ", ".join(sorted(missing_outputs))
                )
        for step in program["steps"]:
            node_id = step["node_id"]
            if node_id in seen_nodes:
                raise ExecutablePackageError(f"duplicate dispatch node ID {node_id}")
            seen_nodes.add(node_id)
            pipeline = _require_reference(
                pipelines,
                step["pipeline_id"],
                f"dispatch {node_id} pipeline",
            )
            expected_bindings = {
                item["binding"] for item in pipeline["descriptor_layout"]
            }
            actual_bindings = {item["binding"] for item in step["resources"]}
            if expected_bindings != actual_bindings:
                raise ExecutablePackageError(
                    f"dispatch {node_id} resources do not match descriptor layout"
                )
            for resource in step["resources"]:
                tensor = _require_reference(
                    tensors,
                    resource["tensor_id"],
                    f"dispatch {node_id} tensor",
                    )
                bindings = program.get("bindings")
                if (
                    bindings is not None
                    and tensor["storage"]["kind"] in {"external", "state"}
                    and resource["tensor_id"]
                    not in (*bindings["inputs"], *bindings["outputs"])
                ):
                    raise ExecutablePackageError(
                        f"dispatch {node_id} exposes external/state tensor "
                        f"{resource['tensor_id']} outside program {program_id} bindings"
                    )
                descriptor = next(
                    item
                    for item in pipeline["descriptor_layout"]
                    if item["binding"] == resource["binding"]
                )
                if tensor["data_type"] != descriptor["data_type"]:
                    raise ExecutablePackageError(
                        f"dispatch {node_id} binding {resource['binding']} data type mismatch"
                    )
            expected_push = {
                item["name"] for item in pipeline["push_constant_layout"]
            }
            if expected_push != set(step["push_constants"]):
                raise ExecutablePackageError(
                    f"dispatch {node_id} push constants do not match pipeline layout"
                )
            push_types = {
                item["name"]: item["data_type"]
                for item in pipeline["push_constant_layout"]
            }
            for name, value in step["push_constants"].items():
                if _push_constant_type(value) != push_types[name]:
                    raise ExecutablePackageError(
                        f"dispatch {node_id} push constant {name} data type mismatch"
                    )
            certificate_id = step.get("certificate_id")
            if certificate_id is not None:
                _require_reference(
                    certificates,
                    certificate_id,
                    f"dispatch {node_id} certificate",
                )
    return manifest


def _mapping(value: Mapping[str, object], name: str) -> dict[str, Any]:
    item = value[name]
    if not isinstance(item, dict):
        raise ExecutablePackageError(f"manifest {name} must be an object")
    return item


def _unique_records(value: Mapping[str, object], name: str) -> dict[str, Any]:
    items = value[name]
    if not isinstance(items, list):
        raise ExecutablePackageError(f"manifest {name} must be an array")
    return _unique_values(items, name)


def _optional_unique_records(
    value: Mapping[str, object], name: str
) -> dict[str, Any]:
    if name not in value:
        return {}
    return _unique_records(value, name)


def _unique_values(items: Sequence[object], name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ExecutablePackageError(f"{name} record is missing an ID")
        item_id = item["id"]
        if item_id in result:
            raise ExecutablePackageError(f"duplicate {name} ID {item_id}")
        result[item_id] = item
    return result


def _require_reference(values: Mapping[str, Any], item_id: str, label: str) -> Any:
    if item_id not in values:
        raise ExecutablePackageError(f"{label} references missing ID {item_id}")
    return values[item_id]


def _artifact_path(root: Path, relative: str, *, require_file: bool) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExecutablePackageError(f"artifact escapes package root: {relative}") from error
    if require_file and not candidate.is_file():
        raise ExecutablePackageError(f"package artifact is missing: {relative}")
    return candidate


def _verified_artifact(root: Path, relative: str, digest: str) -> bytes:
    path = _artifact_path(root, relative, require_file=True)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ExecutablePackageError(f"cannot read package artifact {relative}: {error}") from error
    if _sha256(payload) != digest:
        raise ExecutablePackageError(f"package artifact hash mismatch: {relative}")
    return payload


def _push_constant_type(value: int | float) -> str:
    if isinstance(value, bool):
        raise ExecutablePackageError("boolean push constants are not supported")
    if isinstance(value, float):
        return "float32"
    return "uint32" if value >= 0 else "int32"


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
