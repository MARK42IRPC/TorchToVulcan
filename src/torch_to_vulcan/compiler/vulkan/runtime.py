"""Runtime for executing a validated TTV 0.1 linear package on Vulkan."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..package import ExecutablePackageError, validate_executable_package
from .executor import VulkanExecutionError, VulkanExecutor, _BufferAllocation


_NP_DTYPES: dict[str, np.dtype[Any]] = {
    "BOOL": np.dtype(np.bool_),
    "FLOAT16": np.dtype(np.float16),
    "FLOAT": np.dtype(np.float32),
    "DOUBLE": np.dtype(np.float64),
    "INT8": np.dtype(np.int8),
    "UINT8": np.dtype(np.uint8),
    "INT16": np.dtype(np.int16),
    "UINT16": np.dtype(np.uint16),
    "INT32": np.dtype(np.int32),
    "UINT32": np.dtype(np.uint32),
    "INT64": np.dtype(np.int64),
    "UINT64": np.dtype(np.uint64),
}


@dataclass(frozen=True, slots=True)
class PackageExecutionResult:
    outputs: dict[str, np.ndarray[Any, Any]]
    device_name: str
    elapsed_ms: float
    upload_ms: float = 0.0
    dispatch_ms: float = 0.0
    readback_ms: float = 0.0
    resident: bool = False


@dataclass(frozen=True, slots=True)
class PackageBenchmark:
    iterations: int
    warmup: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    device_name: str
    mean_upload_ms: float = 0.0
    mean_dispatch_ms: float = 0.0
    mean_readback_ms: float = 0.0
    scope: str = "end_to_end"


@dataclass(slots=True)
class _TensorAllocation:
    allocation: _BufferAllocation
    byte_offset: int = 0
    byte_length: int = 0
    host_allocation: _BufferAllocation | None = None


@dataclass(slots=True)
class _PipelineResources:
    shader_module: Any
    descriptor_layout: Any
    pipeline_layout: Any
    pipeline: Any
    push_constant_size: int


class VulkanPackageRuntime(VulkanExecutor):
    """Load and execute one validated static linear TTV package.

    Device buffers, descriptor layouts, pipelines, and the command buffer are
    retained for the lifetime of this object. Calling :meth:`run` therefore
    only uploads external inputs, submits the already-recorded graph, and reads
    declared outputs.
    """

    def __init__(self, directory: str | Path, *, device_local: bool = False) -> None:
        self.package_directory = Path(directory).resolve()
        self.device_local_requested = device_local
        self.device_local_enabled = False
        try:
            self.manifest = validate_executable_package(self.package_directory)
        except ExecutablePackageError:
            raise
        self._shaders: dict[str, bytes] = {}
        self._pipeline_records = {
            item["id"]: item for item in self.manifest["pipelines"]
        }
        self._shader_records = {
            item["id"]: item for item in self.manifest["shaders"]
        }
        self._pipeline_resources: dict[str, _PipelineResources] = {}
        self._tensor_allocations: dict[str, _TensorAllocation] = {}
        self._tensor_templates: dict[str, np.ndarray[Any, Any]] = {}
        self._prepared = False
        self._descriptor_pool: Any = None
        self._command_pool: Any = None
        self._command_buffer: Any = None
        self._command_variants: dict[tuple[bool, bool], Any] = {}
        self._active_recording_command_buffer: Any = None
        self._fence: Any = None
        self._descriptor_sets: list[Any] = []
        self._program_steps = self._load_program_steps()
        self._load_shader_bytes()
        super().__init__()

    @property
    def memory_mode(self) -> str:
        """Return the actual tensor storage mode after the first preparation."""
        return "device-local" if self.device_local_enabled else "host-visible"

    def close(self) -> None:
        if getattr(self, "device", None) is not None:
            try:
                self.vk.vkDeviceWaitIdle(self.device)
            except Exception:
                pass
            if self._fence is not None:
                self.vk.vkDestroyFence(self.device, self._fence, None)
                self._fence = None
            if self._command_pool is not None:
                self.vk.vkDestroyCommandPool(self.device, self._command_pool, None)
                self._command_pool = None
            self._command_variants.clear()
            self._command_buffer = None
            if self._descriptor_pool is not None:
                self.vk.vkDestroyDescriptorPool(self.device, self._descriptor_pool, None)
                self._descriptor_pool = None
            for resources in self._pipeline_resources.values():
                self.vk.vkDestroyPipeline(self.device, resources.pipeline, None)
                self.vk.vkDestroyPipelineLayout(self.device, resources.pipeline_layout, None)
                self.vk.vkDestroyDescriptorSetLayout(
                    self.device, resources.descriptor_layout, None
                )
                self.vk.vkDestroyShaderModule(self.device, resources.shader_module, None)
            self._pipeline_resources.clear()
            released: set[int] = set()
            unmapped: set[int] = set()
            for tensor in self._tensor_allocations.values():
                for allocation in (tensor.allocation, tensor.host_allocation):
                    if allocation is None:
                        continue
                    allocation_id = id(allocation.buffer)
                    if allocation_id not in unmapped:
                        self._unmap_allocation(allocation)
                        unmapped.add(allocation_id)
                    if allocation_id in released:
                        continue
                    released.add(allocation_id)
                    self.vk.vkDestroyBuffer(self.device, allocation.buffer, None)
                    self.vk.vkFreeMemory(self.device, allocation.memory, None)
            self._tensor_allocations.clear()
        super().close()

    def run(
        self,
        inputs: Mapping[str, np.ndarray[Any, Any]] | None,
        *,
        transfer_inputs: bool = True,
        read_outputs: bool = True,
    ) -> PackageExecutionResult:
        """Execute once, optionally measuring resident device buffers only."""
        started = time.perf_counter()
        if inputs is None:
            if not self._prepared:
                raise VulkanExecutionError("首次执行必须提供 package 输入")
        else:
            self._prepare(inputs)
        upload_started = time.perf_counter()
        if transfer_inputs:
            if inputs is None:
                raise VulkanExecutionError("transfer_inputs=True 时必须提供输入")
            self._upload_inputs(inputs)
        upload_ms = (time.perf_counter() - upload_started) * 1000.0
        dispatch_started = time.perf_counter()
        try:
            command_buffer = self._command_variants.get(
                (transfer_inputs, read_outputs),
                self._command_buffer,
            )
            if command_buffer is not None:
                self._submit_command_buffer(command_buffer)
            dispatch_ms = (time.perf_counter() - dispatch_started) * 1000.0
            if read_outputs:
                readback_started = time.perf_counter()
                outputs = {
                    tensor_id: self._read_tensor(tensor_id)
                    for tensor_id in self.manifest["bindings"]["outputs"]
                }
                readback_ms = (time.perf_counter() - readback_started) * 1000.0
            else:
                outputs = {}
                readback_ms = 0.0
        except VulkanExecutionError:
            raise
        except Exception as error:
            raise VulkanExecutionError(str(error) or type(error).__name__) from error
        return PackageExecutionResult(
            outputs,
            self.device_name,
            (time.perf_counter() - started) * 1000.0,
            upload_ms,
            dispatch_ms,
            readback_ms,
            not transfer_inputs and not read_outputs,
        )

    def benchmark(
        self,
        inputs: Mapping[str, np.ndarray[Any, Any]],
        *,
        warmup: int = 3,
        iterations: int = 10,
        resident: bool = False,
    ) -> PackageBenchmark:
        if warmup < 0 or iterations <= 0:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        if resident:
            self._prepare(inputs)
            self._upload_inputs(inputs)
            # Inputs are written to host staging above; perform one transfer
            # before switching to the resident command variant.
            if self.device_local_enabled:
                self._submit_command_buffer(
                    self._command_variants[(True, False)]
                )
            for _ in range(warmup):
                self.run(None, transfer_inputs=False, read_outputs=False)
            samples = [
                self.run(None, transfer_inputs=False, read_outputs=False)
                for _ in range(iterations)
            ]
        else:
            for _ in range(warmup):
                self.run(inputs)
            samples = [self.run(inputs) for _ in range(iterations)]
        values = np.asarray([sample.elapsed_ms for sample in samples], dtype=np.float64)
        return PackageBenchmark(
            iterations,
            warmup,
            float(values.mean()),
            float(np.median(values)),
            float(values.min()),
            float(values.max()),
            self.device_name,
            float(np.mean([sample.upload_ms for sample in samples])),
            float(np.mean([sample.dispatch_ms for sample in samples])),
            float(np.mean([sample.readback_ms for sample in samples])),
            "resident" if resident else "end_to_end",
        )

    def _submit_command_buffer(self, command_buffer: Any) -> None:
        if self._fence is None:
            raise VulkanExecutionError("Vulkan command buffer fence is unavailable")
        self.vk.vkResetFences(self.device, 1, [self._fence])
        submit = self.vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        self.vk.vkQueueSubmit(self.queue, 1, [submit], self._fence)
        result = self.vk.vkWaitForFences(
            self.device,
            1,
            [self._fence],
            self.vk.VK_TRUE,
            10_000_000_000,
        )
        if result not in (None, self.vk.VK_SUCCESS):
            raise VulkanExecutionError(f"等待 Vulkan 推理完成失败: {result}")

    def _load_program_steps(self) -> tuple[dict[str, Any], ...]:
        programs = {
            item["id"]: item for item in self.manifest["programs"]
        }
        program = programs[self.manifest["entry_program"]]
        if program["kind"] != "linear":
            raise ExecutablePackageError(
                f"TTV 0.1 runtime only supports linear programs, got {program['kind']}"
            )
        return tuple(program["steps"])

    def _load_shader_bytes(self) -> None:
        for shader_id, shader in self._shader_records.items():
            self._shaders[shader_id] = (self.package_directory / shader["file"]).read_bytes()

    def _prepare(self, inputs: Mapping[str, np.ndarray[Any, Any]]) -> None:
        if self._prepared:
            self._validate_inputs(inputs)
            return
        expected_inputs = set(self.manifest["bindings"]["inputs"])
        provided_inputs = set(inputs)
        missing = expected_inputs - provided_inputs
        extra = provided_inputs - expected_inputs
        if missing or extra:
            details = []
            if missing:
                details.append(f"缺少输入: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"多余输入: {', '.join(sorted(extra))}")
            raise VulkanExecutionError("；".join(details))

        self.device_local_enabled = (
            self.device_local_requested and self._has_device_local_memory()
        )
        tensors = self.manifest["tensors"]
        input_ids = set(self.manifest["bindings"]["inputs"])
        output_ids = set(self.manifest["bindings"]["outputs"])
        for tensor_id, record in tensors.items():
            storage = record["storage"]
            kind = storage["kind"]
            if kind == "view":
                continue
            shape = tuple(int(value) for value in record["shape"])
            dtype = _dtype(record["data_type"])
            if tensor_id in inputs:
                array = self._validate_input(tensor_id, inputs[tensor_id], record)
            elif kind == "constant":
                array = self._constant_array(tensor_id, record, dtype, shape)
            else:
                array = np.empty(shape, dtype=dtype)
            self._tensor_templates[tensor_id] = array
            allocation = self._allocate_tensor(
                array,
                kind=kind,
                needs_host=tensor_id in input_ids or tensor_id in output_ids,
            )
            self._tensor_allocations[tensor_id] = allocation

        for tensor_id in tensors:
            if tensors[tensor_id]["storage"]["kind"] == "view":
                source_id = tensors[tensor_id]["storage"]["source_tensor"]
                source = self._resolve_allocation(source_id, tensors)
                self._tensor_allocations[tensor_id] = _TensorAllocation(
                    source.allocation,
                    source.byte_offset + int(tensors[tensor_id]["storage"]["byte_offset"]),
                    int(np.prod(tuple(tensors[tensor_id]["shape"])))
                    * _dtype(tensors[tensor_id]["data_type"]).itemsize,
                    (
                        self._create_host_staging(
                            int(np.prod(tuple(tensors[tensor_id]["shape"])))
                            * _dtype(tensors[tensor_id]["data_type"]).itemsize,
                            np.empty(
                                tuple(tensors[tensor_id]["shape"]),
                                dtype=_dtype(tensors[tensor_id]["data_type"]),
                            ),
                        )
                        if self.device_local_enabled
                        and tensor_id in (input_ids | output_ids)
                        else source.host_allocation
                    ),
                )
                self._tensor_templates[tensor_id] = np.empty(
                    tuple(tensors[tensor_id]["shape"]),
                    dtype=_dtype(tensors[tensor_id]["data_type"]),
                )

        self._create_pipeline_resources()
        self._create_program_resources()
        self._prepared = True

    def _validate_inputs(self, inputs: Mapping[str, np.ndarray[Any, Any]]) -> None:
        expected = set(self.manifest["bindings"]["inputs"])
        if set(inputs) != expected:
            raise VulkanExecutionError("输入名称集合与 package manifest 不一致")
        for tensor_id in expected:
            self._validate_input(tensor_id, inputs[tensor_id], self.manifest["tensors"][tensor_id])

    def _validate_input(
        self,
        tensor_id: str,
        value: np.ndarray[Any, Any],
        record: Mapping[str, Any],
    ) -> np.ndarray[Any, Any]:
        array = np.asarray(value)
        expected_dtype = _dtype(record["data_type"])
        expected_shape = tuple(int(item) for item in record["shape"])
        if array.dtype != expected_dtype:
            raise VulkanExecutionError(
                f"输入 {tensor_id} dtype {array.dtype} != {expected_dtype}"
            )
        if array.shape != expected_shape:
            raise VulkanExecutionError(
                f"输入 {tensor_id} shape {array.shape} != {expected_shape}"
            )
        return np.ascontiguousarray(array)

    def _constant_array(
        self,
        tensor_id: str,
        record: Mapping[str, Any],
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
    ) -> np.ndarray[Any, Any]:
        storage = record["storage"]
        blob = self.manifest_blob(storage["blob_id"])
        offset = int(storage["offset"])
        length = int(storage["length"])
        payload = blob[offset : offset + length]
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(payload) != expected:
            raise VulkanExecutionError(
                f"常量 {tensor_id} payload {len(payload)} bytes != expected {expected}"
            )
        return np.frombuffer(payload, dtype=dtype, count=int(np.prod(shape))).copy().reshape(shape)

    def _has_device_local_memory(self) -> bool:
        probe = None
        try:
            probe = self.vk.vkCreateBuffer(
                self.device,
                self.vk.VkBufferCreateInfo(
                    size=4,
                    usage=self.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | self.vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                    | self.vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    sharingMode=self.vk.VK_SHARING_MODE_EXCLUSIVE,
                ),
                None,
            )
            requirements = self.vk.vkGetBufferMemoryRequirements(self.device, probe)
            self._find_device_local_memory_type(requirements.memoryTypeBits)
            return True
        except Exception:
            return False
        finally:
            if probe is not None:
                self.vk.vkDestroyBuffer(self.device, probe, None)

    def _allocate_tensor(
        self,
        array: np.ndarray[Any, Any],
        *,
        kind: str,
        needs_host: bool,
    ) -> _TensorAllocation:
        logical_size = max(array.nbytes, 4)
        if not self.device_local_enabled:
            return _TensorAllocation(
                self._create_buffer(array),
                byte_length=array.nbytes,
            )
        usage = (
            self.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | self.vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | self.vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )
        device = self._create_raw_buffer(
            logical_size,
            usage,
            self.vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        host = None
        if needs_host or kind == "constant":
            host = self._create_host_staging(logical_size, array)
        if kind == "constant" and host is not None and array.nbytes:
            self._copy_buffer(host, device, array.nbytes)
        return _TensorAllocation(device, byte_length=array.nbytes, host_allocation=host)

    def _create_host_staging(
        self,
        size: int,
        initial: np.ndarray[Any, Any],
    ) -> _BufferAllocation:
        allocation = self._create_raw_buffer(
            max(size, 4),
            self.vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | self.vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        mapped = self._map_allocation(allocation)
        if initial.nbytes:
            mapped[: initial.nbytes] = initial.tobytes(order="C")
            self._flush_memory(allocation, 0, initial.nbytes)
        return allocation

    def manifest_blob(self, blob_id: str) -> bytes:
        blobs = {item["id"]: item for item in self.manifest["blobs"]}
        if blob_id not in blobs:
            raise VulkanExecutionError(f"常量引用未知 blob {blob_id}")
        return (self.package_directory / blobs[blob_id]["file"]).read_bytes()

    def _resolve_allocation(
        self,
        tensor_id: str,
        tensors: Mapping[str, Mapping[str, Any]],
    ) -> _TensorAllocation:
        if tensor_id in self._tensor_allocations:
            return self._tensor_allocations[tensor_id]
        record = tensors[tensor_id]
        storage = record["storage"]
        if storage["kind"] != "view":
            raise VulkanExecutionError(f"view source {tensor_id} has no allocation")
        source = self._resolve_allocation(storage["source_tensor"], tensors)
        resolved = _TensorAllocation(
            source.allocation,
            source.byte_offset + int(storage["byte_offset"]),
            int(np.prod(tuple(tensors[tensor_id]["shape"])))
            * _dtype(tensors[tensor_id]["data_type"]).itemsize,
            source.host_allocation,
        )
        self._tensor_allocations[tensor_id] = resolved
        return resolved

    def _upload_inputs(self, inputs: Mapping[str, np.ndarray[Any, Any]]) -> None:
        for tensor_id in self.manifest["bindings"]["inputs"]:
            array = self._validate_input(tensor_id, inputs[tensor_id], self.manifest["tensors"][tensor_id])
            tensor = self._tensor_allocations[tensor_id]
            allocation = tensor.host_allocation or tensor.allocation
            offset = 0 if tensor.host_allocation is not None else tensor.byte_offset
            if offset < 0 or offset + array.nbytes > allocation.size:
                raise VulkanExecutionError(f"输入 {tensor_id} 超出 Vulkan allocation")
            mapped = self._map_allocation(allocation)
            if array.nbytes:
                mapped[offset : offset + array.nbytes] = array.tobytes(order="C")
                self._flush_memory(allocation, offset, array.nbytes)

    def _read_tensor(self, tensor_id: str) -> np.ndarray[Any, Any]:
        record = self.manifest["tensors"][tensor_id]
        template = self._tensor_templates[tensor_id]
        reference = self._tensor_allocations[tensor_id]
        allocation = reference.host_allocation or reference.allocation
        offset = 0 if reference.host_allocation is not None else reference.byte_offset
        if offset < 0 or offset + template.nbytes > allocation.size:
            raise VulkanExecutionError(f"输出 {tensor_id} 超出 Vulkan allocation")
        mapped = self._map_allocation(allocation)
        self._invalidate_memory(
            allocation,
            offset,
            template.nbytes,
        )
        start = offset
        values = np.frombuffer(
            mapped,
            dtype=template.dtype,
            count=template.size,
            offset=start,
        ).copy()
        return values.reshape(tuple(record["shape"]))

    def _create_pipeline_resources(self) -> None:
        for pipeline_id, record in self._pipeline_records.items():
            shader_id = record["shader_id"]
            shader_record = self._shader_records[shader_id]
            shader_module = self._create_shader_module(self._shaders[shader_id])
            descriptors = sorted(record["descriptor_layout"], key=lambda item: item["binding"])
            layout_bindings = [
                self.vk.VkDescriptorSetLayoutBinding(
                    binding=int(item["binding"]),
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for item in descriptors
            ]
            descriptor_layout = self.vk.vkCreateDescriptorSetLayout(
                self.device,
                self.vk.VkDescriptorSetLayoutCreateInfo(
                    bindingCount=len(layout_bindings),
                    pBindings=layout_bindings,
                ),
                None,
            )
            push_size = 4 * len(record["push_constant_layout"])
            push_ranges = []
            if push_size:
                push_ranges.append(
                    self.vk.VkPushConstantRange(
                        stageFlags=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        offset=0,
                        size=push_size,
                    )
                )
            pipeline_layout = self.vk.vkCreatePipelineLayout(
                self.device,
                self.vk.VkPipelineLayoutCreateInfo(
                    setLayoutCount=1,
                    pSetLayouts=[descriptor_layout],
                    pushConstantRangeCount=len(push_ranges),
                    pPushConstantRanges=push_ranges or None,
                ),
                None,
            )
            stage = self.vk.VkPipelineShaderStageCreateInfo(
                stage=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName=shader_record["entry_point"],
            )
            pipeline = self.vk.vkCreateComputePipelines(
                self.device,
                self.vk.VK_NULL_HANDLE,
                1,
                [
                    self.vk.VkComputePipelineCreateInfo(
                        stage=stage,
                        layout=pipeline_layout,
                    )
                ],
                None,
            )[0]
            self._pipeline_resources[pipeline_id] = _PipelineResources(
                shader_module,
                descriptor_layout,
                pipeline_layout,
                pipeline,
                push_size,
            )

    def _create_program_resources(self) -> None:
        if self._program_steps:
            descriptor_count = sum(
                len(self._pipeline_records[step["pipeline_id"]]["descriptor_layout"])
                for step in self._program_steps
            )
            pool_sizes = []
            if descriptor_count:
                pool_sizes.append(
                    self.vk.VkDescriptorPoolSize(
                        type=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=descriptor_count,
                    )
                )
            self._descriptor_pool = self.vk.vkCreateDescriptorPool(
                self.device,
                self.vk.VkDescriptorPoolCreateInfo(
                    maxSets=len(self._program_steps),
                    poolSizeCount=len(pool_sizes),
                    pPoolSizes=pool_sizes or None,
                ),
                None,
            )
            layouts = [
                self._pipeline_resources[step["pipeline_id"]].descriptor_layout
                for step in self._program_steps
            ]
            self._descriptor_sets = list(
                self.vk.vkAllocateDescriptorSets(
                    self.device,
                    self.vk.VkDescriptorSetAllocateInfo(
                        descriptorPool=self._descriptor_pool,
                        descriptorSetCount=len(layouts),
                        pSetLayouts=layouts,
                    ),
                )
            )
            for step, descriptor_set in zip(self._program_steps, self._descriptor_sets, strict=True):
                resources = self._pipeline_records[step["pipeline_id"]]["descriptor_layout"]
                writes = []
                for descriptor in resources:
                    binding = int(descriptor["binding"])
                    resource = next(
                        item for item in step["resources"] if int(item["binding"]) == binding
                    )
                    allocation = self._tensor_allocations[resource["tensor_id"]]
                    descriptor_range = max(4, allocation.byte_length)
                    if (
                        allocation.byte_offset < 0
                        or allocation.byte_offset + descriptor_range
                        > allocation.allocation.size
                    ):
                        raise VulkanExecutionError(
                            f"tensor {resource['tensor_id']} descriptor range exceeds allocation"
                        )
                    buffer_info = self.vk.VkDescriptorBufferInfo(
                        buffer=allocation.allocation.buffer,
                        offset=allocation.byte_offset,
                        range=descriptor_range,
                    )
                    writes.append(
                        self.vk.VkWriteDescriptorSet(
                            dstSet=descriptor_set,
                            dstBinding=binding,
                            descriptorCount=1,
                            descriptorType=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            pBufferInfo=[buffer_info],
                        )
                    )
                if writes:
                    self.vk.vkUpdateDescriptorSets(
                        self.device,
                        len(writes),
                        writes,
                        0,
                        None,
                    )

        # A device-local graph needs a command buffer even when it contains only
        # copies (for example an Identity/Reshape view or a constant output).
        if self._program_steps or self.device_local_enabled:
            self._command_pool = self.vk.vkCreateCommandPool(
                self.device,
                self.vk.VkCommandPoolCreateInfo(queueFamilyIndex=self.queue_family),
                None,
            )
            variant_keys = (
                ((False, False), (False, True), (True, False), (True, True))
                if self.device_local_enabled
                else ((True, True),)
            )
            command_buffers = self.vk.vkAllocateCommandBuffers(
                self.device,
                self.vk.VkCommandBufferAllocateInfo(
                    commandPool=self._command_pool,
                    level=self.vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=len(variant_keys),
                ),
            )
            self._command_variants = dict(zip(variant_keys, command_buffers, strict=True))
            self._command_buffer = self._command_variants[(True, True)]
            for (copy_inputs, copy_outputs), command_buffer in self._command_variants.items():
                self._record_program(
                    command_buffer,
                    copy_inputs=copy_inputs,
                    copy_outputs=copy_outputs,
                )
            self._fence = self.vk.vkCreateFence(self.device, self.vk.VkFenceCreateInfo(), None)

    def _record_program(
        self,
        command_buffer: Any,
        *,
        copy_inputs: bool,
        copy_outputs: bool,
    ) -> None:
        self._active_recording_command_buffer = command_buffer
        self.vk.vkBeginCommandBuffer(
            command_buffer,
            self.vk.VkCommandBufferBeginInfo(
                flags=self.vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
            ),
        )
        copied_inputs = False
        if self.device_local_enabled and copy_inputs:
            for tensor_id in self.manifest["bindings"]["inputs"]:
                tensor = self._tensor_allocations[tensor_id]
                if tensor.host_allocation is None or tensor.byte_length <= 0:
                    continue
                self._record_buffer_copy(
                    tensor.host_allocation,
                    tensor.allocation,
                    tensor.byte_length,
                    destination_offset=tensor.byte_offset,
                )
                copied_inputs = True
            if copied_inputs and self._program_steps:
                self._record_transfer_to_compute_barrier()
            elif copied_inputs:
                self._record_transfer_to_transfer_barrier()

        for index, step in enumerate(self._program_steps):
            pipeline = self._pipeline_resources[step["pipeline_id"]]
            self.vk.vkCmdBindPipeline(
                command_buffer,
                self.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline,
            )
            self.vk.vkCmdBindDescriptorSets(
                command_buffer,
                self.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline_layout,
                0,
                1,
                [self._descriptor_sets[index]],
                0,
                None,
            )
            payload = _pack_push_constants(
                step["push_constants"],
                self._pipeline_records[step["pipeline_id"]]["push_constant_layout"],
            )
            if payload:
                push_value = self.vk.ffi.new("char[]", payload)
                self.vk.vkCmdPushConstants(
                    command_buffer,
                    pipeline.pipeline_layout,
                    self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    len(payload),
                    push_value,
                )
            self.vk.vkCmdDispatch(command_buffer, *step["workgroups"])
            if index + 1 < len(self._program_steps):
                self._record_shader_barrier()
        if self.device_local_enabled:
            copied_outputs = False
            if copy_outputs and self._program_steps:
                output_ids = self.manifest["bindings"]["outputs"]
                if output_ids:
                    self._record_compute_to_transfer_barrier()
            if copy_outputs:
                for tensor_id in self.manifest["bindings"]["outputs"]:
                    tensor = self._tensor_allocations[tensor_id]
                    if tensor.host_allocation is None or tensor.byte_length <= 0:
                        continue
                    self._record_buffer_copy(
                        tensor.allocation,
                        tensor.host_allocation,
                        tensor.byte_length,
                        source_offset=tensor.byte_offset,
                    )
                    copied_outputs = True
            if copied_outputs:
                self._record_transfer_to_host_barrier()
        elif self._program_steps:
            self._record_host_barrier()
        self.vk.vkEndCommandBuffer(command_buffer)
        self._active_recording_command_buffer = None

    def _record_buffer_copy(
        self,
        source: _BufferAllocation,
        destination: _BufferAllocation,
        size: int,
        *,
        source_offset: int = 0,
        destination_offset: int = 0,
    ) -> None:
        if size <= 0:
            return
        if (
            source_offset < 0
            or destination_offset < 0
            or source_offset + size > source.size
            or destination_offset + size > destination.size
        ):
            raise VulkanExecutionError("Vulkan command buffer copy range is outside allocation")
        self.vk.vkCmdCopyBuffer(
            self._active_recording_command_buffer,
            source.buffer,
            destination.buffer,
            1,
            [
                self.vk.VkBufferCopy(
                    srcOffset=source_offset,
                    dstOffset=destination_offset,
                    size=size,
                )
            ],
        )

    def _record_transfer_to_compute_barrier(self) -> None:
        self._record_memory_barrier(
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            self.vk.VK_ACCESS_SHADER_READ_BIT | self.vk.VK_ACCESS_SHADER_WRITE_BIT,
        )

    def _record_compute_to_transfer_barrier(self) -> None:
        self._record_memory_barrier(
            self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            self.vk.VK_ACCESS_SHADER_READ_BIT | self.vk.VK_ACCESS_SHADER_WRITE_BIT,
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_ACCESS_TRANSFER_READ_BIT,
        )

    def _record_transfer_to_transfer_barrier(self) -> None:
        self._record_memory_barrier(
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_ACCESS_TRANSFER_READ_BIT | self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        )

    def _record_transfer_to_host_barrier(self) -> None:
        self._record_memory_barrier(
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            self.vk.VK_PIPELINE_STAGE_HOST_BIT,
            self.vk.VK_ACCESS_HOST_READ_BIT,
        )

    def _record_memory_barrier(
        self,
        source_stage: int,
        source_access: int,
        destination_stage: int,
        destination_access: int,
    ) -> None:
        barrier = self.vk.VkMemoryBarrier(
            srcAccessMask=source_access,
            dstAccessMask=destination_access,
        )
        self.vk.vkCmdPipelineBarrier(
            self._active_recording_command_buffer,
            source_stage,
            destination_stage,
            0,
            1,
            [barrier],
            0,
            None,
            0,
            None,
        )

    def _record_shader_barrier(self) -> None:
        self._record_memory_barrier(
            self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            self.vk.VK_ACCESS_SHADER_READ_BIT | self.vk.VK_ACCESS_SHADER_WRITE_BIT,
            self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            self.vk.VK_ACCESS_SHADER_READ_BIT | self.vk.VK_ACCESS_SHADER_WRITE_BIT,
        )

    def _record_host_barrier(self) -> None:
        self._record_memory_barrier(
            self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            self.vk.VK_ACCESS_SHADER_READ_BIT | self.vk.VK_ACCESS_SHADER_WRITE_BIT,
            self.vk.VK_PIPELINE_STAGE_HOST_BIT,
            self.vk.VK_ACCESS_HOST_READ_BIT,
        )


def _dtype(data_type: str) -> np.dtype[Any]:
    try:
        return _NP_DTYPES[data_type]
    except KeyError as error:
        raise VulkanExecutionError(f"运行时不支持 tensor data type {data_type}") from error


def _pack_push_constants(
    values: Mapping[str, Any],
    layout: list[Mapping[str, Any]],
) -> bytes:
    payload = bytearray()
    for item in layout:
        name = item["name"]
        if name not in values:
            raise VulkanExecutionError(f"缺少 push constant {name}")
        value = values[name]
        data_type = item["data_type"]
        if data_type == "uint32":
            if isinstance(value, bool) or int(value) < 0 or int(value) >= 2**32:
                raise VulkanExecutionError(f"push constant {name} 不是有效 uint32")
            payload.extend(struct.pack("<I", int(value)))
        elif data_type == "int32":
            if isinstance(value, bool) or int(value) < -(2**31) or int(value) >= 2**31:
                raise VulkanExecutionError(f"push constant {name} 不是有效 int32")
            payload.extend(struct.pack("<i", int(value)))
        elif data_type == "float32":
            payload.extend(struct.pack("<f", float(value)))
        else:
            raise VulkanExecutionError(f"未知 push constant 类型 {data_type}")
    return bytes(payload)
