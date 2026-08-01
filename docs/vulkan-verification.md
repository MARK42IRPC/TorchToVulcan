# Vulkan Mapping Verification

## Boundary

Vulkan mapping starts after mathematical semantics have been resolved. Semantic
confidence and kernel verification are independent:

```text
EXACT_RULE / EXACT_FUNCTION
          |
          v
Kernel Candidate -> DispatchPlan -> GLSL -> SPIR-V -> Vulkan execution
                                                       |
ONNX Runtime CPU reference ----------------------------+-> comparison certificate
```

An exact semantic source does not imply that a generated kernel is correct. A
kernel is usable only inside the data type, layout, shape, attribute, and device
range recorded by its verification certificate.

## Initial candidates

The first catalog intentionally accepts a narrow range:

- contiguous FP32 `Add`, `Mul`, `Sub`, and `Div` with ONNX multidirectional
  broadcasting, including scalar inputs;
- contiguous FP32 `Relu`;
- contiguous FP32 `Neg`, `Exp`, `Floor`, `Sin`, `Cos`, `Sqrt`, `Tanh`,
  `Sigmoid`, and `LeakyRelu`;
- specialized row-major FP32 `Transpose` and variable-input `Concat`;
- static-shape `Cast` between FP32 and INT32;
- zero-dispatch `Reshape` and `Identity` view plans.

Every compute shader uses a 256-thread local workgroup, a pushed element count,
and an invocation bounds guard. Broadcast kernels specialize row-major buffer
indices from the output coordinate: unit dimensions read coordinate zero and a
scalar reads element zero. Unsupported types, incompatible ranks/shapes, or arity
are rejected before GLSL generation. Verification inputs are limited to 1,048,576
elements per tensor to keep the interactive audit within a predictable memory
budget.

Symbolic or large compute shapes are converted to small static verification
samples while preserving tensor rank and unit-dimension broadcast axes. The
original target ID and semantic key remain on the certificate, and the audit log
records the original and normalized specifications. This validates the generated
kernel class without allocating a full model-sized tensor; it is not a proof that
the complete graph has been compiled. `UNKNOWN` tensor types are never normalized
or range-verified. A tensor also carries `shape_known`, which distinguishes an
unknown rank from a real scalar shape; unknown ranks are blocked before planning.
In particular, `Reshape` and `Identity` require known matching data types and
ranks before a zero-dispatch certificate can be issued.

The isolated ONNX reference model assigns unique synthetic port names. Original
graphs may reuse names across node ports, while an isolated graph requires unique
definition sites; Vulkan descriptor bindings are positional and remain tied to
the original target specification.

## Verification stages

The runner emits ordered NDJSON events and records the last completed stage:

`GENERATED`, `PLAN_VERIFIED`, `REFERENCE_EXECUTED`, `SPIRV_COMPILED`,
`SPIRV_VALIDATED`, and `DEVICE_VERIFIED` identify how far verification actually
progressed.

1. select a matching Kernel Candidate;
2. generate a `DispatchPlan` and GLSL;
3. synthesize deterministic inputs and execute the single operator with ONNX
   Runtime CPU;
4. compile GLSL with the bundled `@webgpu/glslang` or SDK `glslangValidator`;
5. validate SPIR-V with `spirv-val` when available;
6. execute the dispatch on Vulkan;
7. compare GPU and reference results;
8. write a range/device verification certificate.

Each compute mapping runs three deterministic input cases. Floating-point output
uses `rtol=1e-5` and `atol=1e-6`; integer output must match exactly. A complete
device pass produces `DEVICE_VERIFIED`, records per-case error metrics, and sets
confidence to the passed-case ratio. Zero-dispatch view plans use checked shape,
type, and element-count invariants and produce `RANGE_VERIFIED`.

Missing tools produce `BLOCKED`, not a passing result. Compilation, Vulkan
execution, or numerical comparison errors produce `FAILED`.

## Audit events

`POST /api/verify/stream` accepts deduplicated mapping variants and streams:

- `started`: target count and detected toolchain/device capabilities;
- `log`: timestamped `INFO`, `WARN`, or `ERROR` audit entries;
- `certificate`: the per-target stage, hashes, case counts, and reason;
- `progress`: current target, total targets, percentage, and status;
- `result`: verified, blocked, and failed totals.

The WebUI displays these events in a modal log and keeps the completed result
visible for review.

## Toolchain

The Python verification extra installs ONNX Runtime and the Vulkan loader binding.
Web dependencies include a portable WebAssembly build of Khronos glslang, so a
Vulkan SDK is not required for GLSL compilation. When an SDK installation provides
`glslangValidator` and `spirv-val`, those native tools take precedence. Capability
probing also records the physical device reported by `vulkaninfo --summary`.

## Package runtime and performance boundary

`VulkanPackageRuntime` validates a TTV directory, selects the highest-scored
physical device while penalizing D3D12 compatibility devices, allocates constants
and tensors, creates cached pipelines and descriptor sets, records the complete
linear program with shader-to-shader barriers, and reuses that command buffer on
subsequent calls. The CLI exposes this path as:

```powershell
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --output outputs.npz
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --warmup 3 --iterations 20
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --cpu-onnx model.onnx
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --resident
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --device-local
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --device-local --resident
```

The runtime defaults to host-visible storage buffers. `--device-local` switches
materialized tensors to device-local buffers and uses host-visible staging for
external inputs and declared outputs. The recorded program places explicit
transfer-to-compute, compute-to-transfer, transfer-to-host, and inter-dispatch
memory barriers. Four cached command variants cover input/output transfer
combinations, so `--resident` can measure dispatch with all tensors already on
the device. If device-local memory is unavailable, the runtime falls back to
host-visible storage and the CLI reports the actual mode.

Before claiming that a model is faster than CPU, benchmark at least these
separate scopes:

1. CPU ONNX Runtime end-to-end latency;
2. GPU upload + dispatch + readback latency;
3. GPU steady-state dispatch latency with input/output buffers resident;
4. one-time package load and pipeline creation cost.

Small elementwise graphs are usually dominated by queue submission, command
buffer synchronization, and host copies. Whole-model speed requires device-local
arena allocation with staging buffers, tensor lifetime aliasing, fewer barriers,
operator fusion, tiled kernels for memory reuse, and asynchronous double-buffered
execution. A passing mapping certificate proves numerical equivalence for its
tested range; it does not imply a whole-model speedup.
