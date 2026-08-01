# TTV Executable Package 0.1

## Purpose

A TTV executable package is the durable output of compilation. It does not store
one shader for every ONNX node. It stores deduplicated executable artifacts and a
program that binds model tensors to those artifacts.

```text
ONNX -> typed TTV IR -> compile-time evaluation -> Vulkan lowering
     -> memory plan -> executable package -> runtime
```

Version 0.1 is a directory package. A future archive container may use the same
manifest without changing its internal paths.

## Package layout

```text
model.ttv/
|-- manifest.json
|-- constants/
|   `-- weights.bin
|-- shaders/
|   `-- <sha256>.spv
|-- certificates/
|   `-- kernels.json
`-- debug/
    `-- shaders/<sha256>.comp
```

Only `manifest.json`, referenced constant blobs, and referenced SPIR-V modules are
required for execution. Certificates and GLSL sources are audit artifacts.

## Operator materialization

Operators have three different persistent forms:

1. Compile-time operators such as `Constant` and shape-only `Gather` are evaluated
   by the compiler. Their result is stored in `constants/weights.bin`; the operator
   does not appear as a runtime dispatch.
2. View operators such as a valid `Reshape` create tensor storage aliases with a
   new shape/stride description. They do not own a shader.
3. Runtime operators lower to one or more program steps. A dispatch step references
   a pipeline, tensors, workgroup counts, and push constant values.

An operator is not materialized merely because pseudocode or a verification sample
exists. Materialization requires a concrete tensor specification, selected kernel,
compiled SPIR-V, resource bindings, and a package integrity check.

## Artifact identity

SPIR-V modules are named by the SHA-256 hash of their bytes. Identical specialized
kernels are stored once and may be referenced by many pipelines or dispatches.
The manifest also records the hash of every external blob. Package validation
recomputes those hashes before execution.

Constants are appended to one binary blob with 256-byte aligned offsets. Tensor
records retain dtype, shape, offset, and byte length. JSON never contains weight
arrays.

## Programs and control flow

Version 0.1 implements a `linear` program containing ordered dispatch steps.
The manifest reserves named programs so later versions can reference loop and
branch bodies without unrolling them. Autoregressive `Loop` bodies will therefore
be stored as subprograms that reuse pipelines and carried tensor slots.

The package stores SPIR-V, not live Vulkan objects. Descriptor sets, command
buffers, device memory, and pipelines are created by the runtime. Driver-specific
`VkPipelineCache` data belongs in a local cache keyed by physical device, driver,
and shader hash; it is not part of the portable model package.

## Dynamic shapes

Normalized verification samples are never execution profiles. Version 0.1 only
materializes concrete static tensor shapes. A dynamic model must be compiled for
one or more explicit profiles, for example `batch=1, token_count=128`. Runtime
specialization and bounded symbolic dimensions are later format capabilities.

## Certificates

A dispatch may reference a mapping certificate containing the semantic key,
kernel ID, shader hash, tested device, tolerances, and passed case count. A
certificate is evidence for a tested range, not a formal proof and not a runtime
dependency.

## Validation boundary

Package validation rejects:

- a manifest that does not match `model-package.schema.json`;
- absolute paths or paths escaping the package directory;
- missing or hash-mismatched blobs and shaders;
- duplicate artifact IDs;
- dangling shader, pipeline, tensor, program, or certificate references;
- constant tensor ranges outside their blob or violating alignment;
- dispatch resources that do not match the pipeline descriptor layout.

Version 0.1 implementation proceeds in slices:

1. write and reload static linear programs, SPIR-V, constants, and certificates;
2. evaluate and persist constant/shape subgraphs;
3. add tensor arena allocation and storage aliases;
4. load and execute a complete linear package;
5. add named loop/branch subprograms and shape profiles.

## Current implementation

Slices 1 and 2 are implemented, including the first static root-graph compiler
coordinator. The
constant evaluator uses ONNX reference implementations for a strict standard
operator allowlist and directly decodes `Constant` TensorProto attributes to
preserve scalar rank. It currently propagates:

```text
Constant, Shape, Size, Cast, Identity, Unsqueeze, Squeeze, Reshape,
Slice, Gather, Concat, Add, Sub, Mul, Div
```

Evaluation requires every data input to be known. `Shape` and `Size` may also
fold a runtime tensor whose declared shape is completely static. Unsupported
domains, dynamic dimensions, oversized values, strings, and evaluation errors
remain runtime work and produce diagnostics instead of aborting compilation.

Only constants crossing from the folded region into a runtime node, plus folded
graph outputs, are materialized. Internal compile-time intermediates are omitted
from `weights.bin`. The corresponding graph rewrite removes folded root-graph
nodes and inserts those boundary values as ONNX initializers, allowing the
remaining graph to be checked and reference-executed before Vulkan lowering.

The static compiler then runs strict ONNX type/shape inference over the rewritten
graph, materializes referenced initializers, selects a registered kernel for every
remaining node, compiles GLSL to SPIR-V, and appends dispatches in graph order.
`Identity` and valid `Reshape` nodes become storage views. Graph inputs and graph
outputs retain their real declared shapes; verification-only normalized shapes
are never used for package compilation. Large linear kernels use a two-dimensional
workgroup grid flattened by the shader.

Compilation is all-or-nothing. Dynamic shapes, `If`/`Loop`/`Scan`, nested graph
attributes, missing tensor metadata, unsupported data types, and unregistered
kernels produce node-indexed diagnostics before the destination directory is
created. The Python Vulkan package runtime now loads and executes a complete
`linear` program. It keeps one device, one allocation per materialized tensor,
cached pipelines, descriptor sets, and a recorded command buffer alive across
calls. The runtime accepts an `.npz` whose keys are the manifest input tensor
IDs:

```powershell
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --output outputs.npz
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --warmup 3 --iterations 20
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --cpu-onnx model.onnx
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --device-local
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --device-local --resident
```

The runtime defaults to host-visible buffers so correctness and portability are
available on more Vulkan implementations. `--device-local` allocates tensor
buffers in device-local memory and records staging upload/readback copies plus
the required transfer/compute/host memory barriers. It keeps four command
variants so resident calls can omit both external transfers. If no suitable
device-local memory type exists, the runtime falls back to host-visible storage;
the CLI prints both the requested and actual memory modes. This is a real
execution path, but it is not by itself a CPU-speed guarantee: large models
still need tensor lifetime aliasing, pipeline-cache persistence, operator
fusion, tiled kernels, and asynchronous double buffering.

Benchmarks report upload, dispatch, readback, warmup, and steady-state timings
separately. A tiny operator can be dominated by Vulkan submission overhead and
does not establish whole-model CPU speedup.
`--cpu-onnx` runs the same input through ONNX Runtime CPU and reports
`CPU latency / GPU latency`; a value above `1.0x` means the measured GPU path is
faster for that exact input and timing scope.

Compile a supported static ONNX model and validate the resulting directory:

```powershell
.venv\Scripts\ttv compile model.onnx artifacts\model.ttv
.venv\Scripts\ttv validate-package artifacts\model.ttv
```

Use `--no-debug-sources` when the human-readable GLSL audit files are not needed.

Validate a materialized directory without starting the Web service:

```powershell
.venv\Scripts\ttv validate-package artifacts\smoke-add.ttv
```
