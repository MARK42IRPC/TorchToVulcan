"""Minimal Vulkan compute executor for verification dispatches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .ir import DispatchStep


class VulkanExecutionError(RuntimeError):
    """A generated dispatch could not be executed by Vulkan."""


@dataclass(frozen=True, slots=True)
class VulkanExecutionResult:
    outputs: tuple[np.ndarray[Any, Any], ...]
    device_name: str


@dataclass(slots=True)
class _BufferAllocation:
    buffer: Any
    memory: Any
    size: int
    memory_flags: int = 0
    non_coherent_atom_size: int = 1
    mapped: Any = None


class VulkanExecutor:
    def __init__(self) -> None:
        try:
            import vulkan as vk
        except ImportError as error:
            raise VulkanExecutionError("Vulkan Python 绑定不可用") from error

        self.vk = vk
        self.instance: Any = None
        self.device: Any = None
        self.queue: Any = None
        self.physical_device: Any = None
        self.queue_family = -1
        self.device_name = "UNKNOWN"
        try:
            self._open()
        except Exception as error:
            self.close()
            raise VulkanExecutionError(f"初始化 Vulkan 设备失败: {error}") from error

    def __enter__(self) -> "VulkanExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.device is not None:
            try:
                self.vk.vkDeviceWaitIdle(self.device)
            finally:
                self.vk.vkDestroyDevice(self.device, None)
                self.device = None
        if self.instance is not None:
            self.vk.vkDestroyInstance(self.instance, None)
            self.instance = None

    def execute(
        self,
        step: DispatchStep,
        spirv: bytes,
        inputs: Sequence[np.ndarray[Any, Any]],
        output_templates: Sequence[np.ndarray[Any, Any]],
    ) -> VulkanExecutionResult:
        arrays = [np.ascontiguousarray(value) for value in inputs]
        arrays.extend(np.empty_like(value) for value in output_templates)
        if len(arrays) != len(step.bindings):
            raise VulkanExecutionError(
                f"Buffer 数量不匹配: 计划 {len(step.bindings)}，实际 {len(arrays)}"
            )

        allocations: list[_BufferAllocation] = []
        shader_module = None
        descriptor_layout = None
        pipeline_layout = None
        pipeline = None
        descriptor_pool = None
        command_pool = None
        fence = None
        try:
            allocations = [self._create_buffer(array) for array in arrays]
            shader_module = self._create_shader_module(spirv)
            descriptor_layout = self._create_descriptor_layout(step)
            pipeline_layout = self._create_pipeline_layout(descriptor_layout)
            pipeline = self._create_pipeline(shader_module, pipeline_layout)
            descriptor_pool, descriptor_set = self._create_descriptor_set(
                step,
                descriptor_layout,
                allocations,
            )
            command_pool, command_buffer = self._record_dispatch(
                step,
                pipeline,
                pipeline_layout,
                descriptor_set,
            )
            fence = self.vk.vkCreateFence(
                self.device,
                self.vk.VkFenceCreateInfo(),
                None,
            )
            submit = self.vk.VkSubmitInfo(
                commandBufferCount=1,
                pCommandBuffers=[command_buffer],
            )
            self.vk.vkQueueSubmit(self.queue, 1, [submit], fence)
            self.vk.vkWaitForFences(
                self.device,
                1,
                [fence],
                self.vk.VK_TRUE,
                10_000_000_000,
            )
            output_offset = len(inputs)
            outputs = tuple(
                self._read_buffer(allocations[output_offset + index], template)
                for index, template in enumerate(output_templates)
            )
            return VulkanExecutionResult(outputs, self.device_name)
        except VulkanExecutionError:
            raise
        except Exception as error:
            raise VulkanExecutionError(str(error) or type(error).__name__) from error
        finally:
            if fence is not None:
                self.vk.vkDestroyFence(self.device, fence, None)
            if command_pool is not None:
                self.vk.vkDestroyCommandPool(self.device, command_pool, None)
            if descriptor_pool is not None:
                self.vk.vkDestroyDescriptorPool(self.device, descriptor_pool, None)
            if pipeline is not None:
                self.vk.vkDestroyPipeline(self.device, pipeline, None)
            if pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
            if descriptor_layout is not None:
                self.vk.vkDestroyDescriptorSetLayout(self.device, descriptor_layout, None)
            if shader_module is not None:
                self.vk.vkDestroyShaderModule(self.device, shader_module, None)
            for allocation in reversed(allocations):
                self._unmap_allocation(allocation)
                self.vk.vkDestroyBuffer(self.device, allocation.buffer, None)
                self.vk.vkFreeMemory(self.device, allocation.memory, None)

    def _open(self) -> None:
        app_info = self.vk.VkApplicationInfo(
            pApplicationName="TorchToVulcan",
            applicationVersion=1,
            pEngineName="TorchToVulcanVerifier",
            engineVersion=1,
            apiVersion=self.vk.VK_API_VERSION_1_0,
        )
        self.instance = self.vk.vkCreateInstance(
            self.vk.VkInstanceCreateInfo(pApplicationInfo=app_info),
            None,
        )
        devices = list(self.vk.vkEnumeratePhysicalDevices(self.instance))
        if not devices:
            raise VulkanExecutionError("未发现 Vulkan 物理设备")
        self.physical_device = max(devices, key=self._device_score)
        properties = self.vk.vkGetPhysicalDeviceProperties(self.physical_device)
        self.device_name = str(properties.deviceName)
        queue_families = self.vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self.physical_device
        )
        compute_families = [
            index
            for index, properties in enumerate(queue_families)
            if properties.queueFlags & self.vk.VK_QUEUE_COMPUTE_BIT
        ]
        if not compute_families:
            raise VulkanExecutionError("设备没有 Vulkan Compute Queue")
        self.queue_family = min(
            compute_families,
            key=lambda index: bool(
                queue_families[index].queueFlags & self.vk.VK_QUEUE_GRAPHICS_BIT
            ),
        )
        queue_info = self.vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        self.device = self.vk.vkCreateDevice(
            self.physical_device,
            self.vk.VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=[queue_info],
            ),
            None,
        )
        self.queue = self.vk.vkGetDeviceQueue(self.device, self.queue_family, 0)

    def _device_score(self, device: Any) -> int:
        properties = self.vk.vkGetPhysicalDeviceProperties(device)
        score = {
            self.vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 100,
            self.vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 50,
        }.get(properties.deviceType, 0)
        if str(properties.deviceName).startswith("Microsoft Direct3D12"):
            score -= 80
        return score

    def _create_buffer(
        self,
        array: np.ndarray[Any, Any],
    ) -> _BufferAllocation:
        size = max(array.nbytes, 4)
        allocation = self._create_raw_buffer(
            size,
            self.vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        mapped = self._map_allocation(allocation)
        if array.nbytes:
            mapped[: array.nbytes] = array.tobytes()
            self._flush_memory(allocation, 0, array.nbytes)
        return allocation

    def _create_raw_buffer(
        self,
        size: int,
        usage: int,
        required_flags: int,
        preferred_flags: int = 0,
    ) -> _BufferAllocation:
        if size <= 0:
            raise VulkanExecutionError("Vulkan buffer size must be positive")
        buffer = self.vk.vkCreateBuffer(
            self.device,
            self.vk.VkBufferCreateInfo(
                size=size,
                usage=usage,
                sharingMode=self.vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        try:
            requirements = self.vk.vkGetBufferMemoryRequirements(self.device, buffer)
            memory_type, memory_flags = self._find_memory_type_with_preferences(
                requirements.memoryTypeBits,
                required_flags,
                preferred_flags,
            )
            memory = self.vk.vkAllocateMemory(
                self.device,
                self.vk.VkMemoryAllocateInfo(
                    allocationSize=requirements.size,
                    memoryTypeIndex=memory_type,
                ),
                None,
            )
            self.vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            properties = self.vk.vkGetPhysicalDeviceProperties(self.physical_device)
            return _BufferAllocation(
                buffer,
                memory,
                int(requirements.size),
                memory_flags,
                int(properties.limits.nonCoherentAtomSize),
            )
        except Exception:
            self.vk.vkDestroyBuffer(self.device, buffer, None)
            raise

    def _copy_buffer(
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
            raise VulkanExecutionError("Vulkan buffer copy range is outside allocation")
        command_pool = None
        fence = None
        try:
            command_pool = self.vk.vkCreateCommandPool(
                self.device,
                self.vk.VkCommandPoolCreateInfo(queueFamilyIndex=self.queue_family),
                None,
            )
            command_buffer = self.vk.vkAllocateCommandBuffers(
                self.device,
                self.vk.VkCommandBufferAllocateInfo(
                    commandPool=command_pool,
                    level=self.vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
            )[0]
            self.vk.vkBeginCommandBuffer(
                command_buffer,
                self.vk.VkCommandBufferBeginInfo(
                    flags=self.vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                ),
            )
            self.vk.vkCmdCopyBuffer(
                command_buffer,
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
            self.vk.vkEndCommandBuffer(command_buffer)
            fence = self.vk.vkCreateFence(
                self.device,
                self.vk.VkFenceCreateInfo(),
                None,
            )
            submit = self.vk.VkSubmitInfo(
                commandBufferCount=1,
                pCommandBuffers=[command_buffer],
            )
            self.vk.vkQueueSubmit(self.queue, 1, [submit], fence)
            result = self.vk.vkWaitForFences(
                self.device,
                1,
                [fence],
                self.vk.VK_TRUE,
                10_000_000_000,
            )
            if result not in (None, self.vk.VK_SUCCESS):
                raise VulkanExecutionError(f"Vulkan buffer copy failed: {result}")
        except VulkanExecutionError:
            raise
        except Exception as error:
            raise VulkanExecutionError(str(error) or type(error).__name__) from error
        finally:
            if fence is not None:
                self.vk.vkDestroyFence(self.device, fence, None)
            if command_pool is not None:
                self.vk.vkDestroyCommandPool(self.device, command_pool, None)

    def _read_buffer(
        self,
        allocation: _BufferAllocation,
        template: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        mapped = self._map_allocation(allocation)
        if not self._is_host_coherent(allocation):
            self._invalidate_memory(allocation, 0, allocation.size)
        return np.frombuffer(
            mapped,
            dtype=template.dtype,
            count=template.size,
        ).copy().reshape(template.shape)

    def _map_allocation(self, allocation: _BufferAllocation) -> Any:
        if allocation.mapped is None:
            allocation.mapped = self.vk.vkMapMemory(
                self.device,
                allocation.memory,
                0,
                allocation.size,
                0,
            )
        return allocation.mapped

    def _unmap_allocation(self, allocation: _BufferAllocation) -> None:
        if allocation.mapped is not None:
            self.vk.vkUnmapMemory(self.device, allocation.memory)
            allocation.mapped = None

    def _find_memory_type(self, type_bits: int, flags: int) -> int:
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for index in range(properties.memoryTypeCount):
            supported = type_bits & (1 << index)
            available = properties.memoryTypes[index].propertyFlags
            if supported and available & flags == flags:
                return index
        raise VulkanExecutionError("未找到 HOST_VISIBLE | HOST_COHERENT 内存类型")

    def _find_host_memory_type(self, type_bits: int) -> tuple[int, int]:
        return self._find_memory_type_with_preferences(
            type_bits,
            self.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def _find_device_local_memory_type(self, type_bits: int) -> tuple[int, int]:
        return self._find_memory_type_with_preferences(
            type_bits,
            self.vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,
        )

    def _find_memory_type_with_preferences(
        self,
        type_bits: int,
        required_flags: int,
        preferred_flags: int,
    ) -> tuple[int, int]:
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        fallback: tuple[int, int] | None = None
        for index in range(properties.memoryTypeCount):
            if not type_bits & (1 << index):
                continue
            flags = properties.memoryTypes[index].propertyFlags
            if flags & required_flags != required_flags:
                continue
            if flags & preferred_flags == preferred_flags:
                return index, flags
            if fallback is None:
                fallback = index, flags
        if fallback is not None:
            return fallback
        raise VulkanExecutionError(
            f"未找到满足 Vulkan memory flags 0x{required_flags:x} 的内存类型"
        )

    def _is_host_coherent(self, allocation: _BufferAllocation) -> bool:
        return bool(
            allocation.memory_flags & self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )

    def _flush_memory(
        self,
        allocation: _BufferAllocation,
        offset: int,
        size: int,
    ) -> None:
        if size <= 0 or self._is_host_coherent(allocation):
            return
        aligned_offset, aligned_size = self._aligned_mapped_range(offset, size, allocation)
        self.vk.vkFlushMappedMemoryRanges(
            self.device,
            1,
            [
                self.vk.VkMappedMemoryRange(
                    memory=allocation.memory,
                    offset=aligned_offset,
                    size=aligned_size,
                )
            ],
        )

    def _invalidate_memory(
        self,
        allocation: _BufferAllocation,
        offset: int,
        size: int,
    ) -> None:
        if size <= 0 or self._is_host_coherent(allocation):
            return
        aligned_offset, aligned_size = self._aligned_mapped_range(offset, size, allocation)
        self.vk.vkInvalidateMappedMemoryRanges(
            self.device,
            1,
            [
                self.vk.VkMappedMemoryRange(
                    memory=allocation.memory,
                    offset=aligned_offset,
                    size=aligned_size,
                )
            ],
        )

    def _aligned_mapped_range(
        self,
        offset: int,
        size: int,
        allocation: _BufferAllocation,
    ) -> tuple[int, int]:
        atom = max(1, allocation.non_coherent_atom_size)
        aligned_offset = offset // atom * atom
        end = min(allocation.size, ((offset + size + atom - 1) // atom) * atom)
        return aligned_offset, max(1, end - aligned_offset)

    def _create_shader_module(self, spirv: bytes) -> Any:
        if len(spirv) % 4:
            raise VulkanExecutionError("SPIR-V 字节数必须为 4 的倍数")
        return self.vk.vkCreateShaderModule(
            self.device,
            self.vk.VkShaderModuleCreateInfo(codeSize=len(spirv), pCode=spirv),
            None,
        )

    def _create_descriptor_layout(self, step: DispatchStep) -> Any:
        bindings = [
            self.vk.VkDescriptorSetLayoutBinding(
                binding=binding.binding,
                descriptorType=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for binding in step.bindings
        ]
        return self.vk.vkCreateDescriptorSetLayout(
            self.device,
            self.vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings),
                pBindings=bindings,
            ),
            None,
        )

    def _create_pipeline_layout(self, descriptor_layout: Any) -> Any:
        push_range = self.vk.VkPushConstantRange(
            stageFlags=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=4,
        )
        return self.vk.vkCreatePipelineLayout(
            self.device,
            self.vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=1,
                pSetLayouts=[descriptor_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_range],
            ),
            None,
        )

    def _create_pipeline(self, shader_module: Any, pipeline_layout: Any) -> Any:
        stage = self.vk.VkPipelineShaderStageCreateInfo(
            stage=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )
        pipelines = self.vk.vkCreateComputePipelines(
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
        )
        return pipelines[0]

    def _create_descriptor_set(
        self,
        step: DispatchStep,
        descriptor_layout: Any,
        allocations: Sequence[_BufferAllocation],
    ) -> tuple[Any, Any]:
        pool = self.vk.vkCreateDescriptorPool(
            self.device,
            self.vk.VkDescriptorPoolCreateInfo(
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[
                    self.vk.VkDescriptorPoolSize(
                        type=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=len(allocations),
                    )
                ],
            ),
            None,
        )
        descriptor_set = self.vk.vkAllocateDescriptorSets(
            self.device,
            self.vk.VkDescriptorSetAllocateInfo(
                descriptorPool=pool,
                descriptorSetCount=1,
                pSetLayouts=[descriptor_layout],
            ),
        )[0]
        buffer_infos = [
            self.vk.VkDescriptorBufferInfo(
                buffer=allocation.buffer,
                offset=0,
                range=allocation.size,
            )
            for allocation in allocations
        ]
        writes = [
            self.vk.VkWriteDescriptorSet(
                dstSet=descriptor_set,
                dstBinding=step.bindings[index].binding,
                descriptorCount=1,
                descriptorType=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buffer_infos[index]],
            )
            for index in range(len(allocations))
        ]
        self.vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        return pool, descriptor_set

    def _record_dispatch(
        self,
        step: DispatchStep,
        pipeline: Any,
        pipeline_layout: Any,
        descriptor_set: Any,
    ) -> tuple[Any, Any]:
        pool = self.vk.vkCreateCommandPool(
            self.device,
            self.vk.VkCommandPoolCreateInfo(queueFamilyIndex=self.queue_family),
            None,
        )
        command_buffer = self.vk.vkAllocateCommandBuffers(
            self.device,
            self.vk.VkCommandBufferAllocateInfo(
                commandPool=pool,
                level=self.vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        self.vk.vkBeginCommandBuffer(
            command_buffer,
            self.vk.VkCommandBufferBeginInfo(
                flags=self.vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            ),
        )
        self.vk.vkCmdBindPipeline(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline,
        )
        self.vk.vkCmdBindDescriptorSets(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
        )
        element_count = int(step.push_constants.get("element_count", 0))
        push_value = self.vk.ffi.new("uint32_t *", element_count)
        self.vk.vkCmdPushConstants(
            command_buffer,
            pipeline_layout,
            self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            4,
            push_value,
        )
        self.vk.vkCmdDispatch(command_buffer, *step.workgroups)
        barrier = self.vk.VkMemoryBarrier(
            srcAccessMask=self.vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=self.vk.VK_ACCESS_HOST_READ_BIT,
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            self.vk.VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            [barrier],
            0,
            None,
            0,
            None,
        )
        self.vk.vkEndCommandBuffer(command_buffer)
        return pool, command_buffer
