# ONNX to Vulkan Coverage Architecture

## Purpose

The long-term compiler target is broad ONNX coverage with explicit limits. It
is not realistic to promise that every ONNX graph will compile to every Vulkan
device. The compiler must instead make three things precise:

1. which ONNX semantics are understood;
2. which Vulkan implementation can lower those semantics;
3. which shape, type, layout, and device range has been verified.

An unsupported graph must fail with node-local diagnostics. It must not silently
execute part of the graph on CPU or claim Vulkan execution when only a semantic
description exists.

## Target pipeline

```text
ONNX + external data
        |
        v
ONNX adapter and version normalization
        |
        v
Normalized Graph IR
        |
        +--> shape/type/profile analysis
        +--> constant folding and graph decomposition
        +--> layout and memory planning
        |
        v
TTV semantic and schedule IR
        |
        v
Vulkan capability selection
        |
        v
Dispatch programs + SPIR-V + certificates
        |
        v
Vulkan runtime
```

The ONNX adapter is responsible for protobuf details, opset rules, function
expansion, external tensors, and nested graphs. The backend must consume typed
compiler IR rather than raw ONNX protobuf objects.

## Support is layered

An operator is not simply supported or unsupported. The compiler tracks these
independent levels:

| Level | Meaning |
| --- | --- |
| Semantic | The ONNX version and attributes have a defined mathematical meaning. |
| Normalized | The operator has been decomposed into the internal IR and its shape/type rules pass. |
| Lowered | A backend kernel candidate can represent the normalized operation. |
| Compiled | The generated shader passes SPIR-V validation. |
| Device verified | The dispatch matches a CPU reference over a declared profile and device range. |

The UI and reports should show the highest achieved level and the reason for
the next blocked level. Semantic support alone must never be shown as Vulkan
executability.

## Stable compiler contracts

`compiler/contracts.py` owns contracts shared by frontends and backends:

- `ShapeProfile` binds symbolic ONNX dimensions to one concrete compilation
  profile;
- `OperatorCapability` describes opset, dtype, layout, and control-flow limits
  for one kernel candidate;
- `BackendCapabilities` exposes backend-wide limits and the operator matrix.

The current TTV 0.1 package is still static, but the compiler API now accepts a
profile boundary. A model with `batch` or `sequence` dimensions can be compiled
when the caller supplies a JSON profile such as:

```json
{
  "name": "batch1-sequence128",
  "dimensions": {
    "batch": 1,
    "sequence": 128
  }
}
```

This is a specialization, not general runtime dynamic-shape support. The
package must record the selected profile before profile sets and runtime shape
dispatch are introduced.

## IR boundaries

The normalized IR must represent the following explicitly:

- tensor dtype, rank, symbolic dimensions, strides, and layout;
- views and storage aliases;
- optional ONNX inputs and variadic inputs;
- initializer storage and external-data provenance;
- subgraphs, branches, loops, and carried state;
- quantization scale, zero point, axis, block size, and accumulation dtype;
- node source location, domain, opset, and normalized attributes.

The first implementation of this boundary lives in `compiler/onnx/ir.py`.
It currently materializes `NormalizedModel`, `NormalizedGraph`,
`NormalizedNode`, `NormalizedTensor`, and constant records, including nested
graphs, symbolic dimensions, contiguous row-major strides, and profile
specialization. The package compiler consumes this IR instead of reading ONNX
protobuf tensor records directly. Unknown dimensions still block the static
TTV 0.1 package, by design.

The Vulkan schedule IR must additionally represent:

- descriptor bindings and access modes;
- push constants and specialization constants;
- dispatch dimensions and runtime bounds;
- barriers and transfer operations;
- tensor lifetimes and allocation aliases;
- reusable subprograms and host-driven call points.

The existing `linear` program remains a compatibility format for the first
static slice. The 0.1 manifest now also accepts named `subprogram` records,
explicit profile records, persistent state metadata, and host-loop records.
These records make call boundaries and lifetimes explicit without changing the
old `main/linear` reader. They do not yet provide general device-side control
flow or dynamic tensor allocation.

## Dynamic shapes and control flow

The implementation order is deliberately conservative:

1. static graphs;
2. symbolic graphs specialized by `ShapeProfile`;
3. bounded profiles with runtime shape checks;
4. host-driven `If` and `Loop` subprograms;
5. device-side control flow where it is demonstrably useful.

The host-driven stage is sufficient for the first autoregressive integration
slice. A host loop can invoke a Vulkan subprogram, read a small stop flag or
token, and invoke the next iteration while persistent state buffers stay on the
same device. It may be slower, but it keeps control semantics explicit while
tensor computation remains on Vulkan. Full KV-cache append and sampling still
need a model-level contract and kernels.

## Dtype and quantization policy

The backend must distinguish storage dtype from compute dtype. Quantized models
cannot be treated as ordinary byte buffers. The normalized representation must
preserve:

- INT8/UINT8 storage type;
- scale and zero point tensor or scalar;
- per-tensor, per-channel, and block quantization;
- signedness and saturation behavior;
- integer accumulation type;
- packed INT4 layout and block size.

The first general implementation may legally dequantize weights during
compilation and run FP32 kernels. This establishes correctness and Vulkan
execution before native INT8/INT4 kernels are optimized.

## Operator implementation order

The coverage matrix should grow by families, with reference tests for every
new family:

1. views, casts, elementwise arithmetic, comparisons, broadcast, and shape ops;
2. reductions, Softmax, normalization, and indexing;
3. MatMul/Gemm and attention building blocks;
4. Conv, ConvTranspose, pooling, and audio transforms;
5. quantization and packed-weight operators;
6. sequences, branches, loops, and model-specific custom domains.

An operator entry must include ONNX version rules, shape/type inference,
normalization, at least one backend candidate, invalid-input tests, and CPU/GPU
differential verification.

Current backend baseline: FP32 `MatMul` and `Gemm` are lowered to guarded
Vulkan compute shaders and can be materialized into the existing linear
package. `MatMul` covers two-dimensional matrices and statically shaped
rank-at-least-two inputs with trailing batch broadcast. `Gemm` remains
two-dimensional and covers `transA`, `transB`, finite
`alpha`/`beta`, and scalar/vector/matrix C broadcast forms. FP16/INT8/INT4,
runtime dynamic shapes, and native quantized execution remain outside this
baseline.

The first static Transformer acceptance slice is now device-verified for
batched MatMul, `ReduceMean`, `Softmax`, `LayerNormalization`, and their
composition in the linear TTV package. The current implementation order is:

1. KV-cache token append, position bounds, and attention cache layout;
2. INT8 dequantization and offline INT4 weight dequantization;
3. native quantized kernels and broader Transformer/Aemeath subgraphs.

The Transformer sequence is an engineering acceptance path, not a promise of
general Transformer or Aemeath support. Each step must pass normalization,
shape/type validation, kernel lowering, SPIR-V compilation, real Vulkan
execution, and reference differential verification before the matrix is
expanded.

## Aemeath as a coverage milestone

The Aemeath package is a useful complex-model milestone, not the definition of
the generic compiler. It exercises:

- dynamic sequence and audio lengths;
- Transformer MatMul/Gemm, LayerNorm, Softmax, TopK, and KV cache;
- INT4 `com.microsoft::MatMulNBits`;
- INT8 `MatMulInteger` and dynamic quantization;
- Conv/ConvTranspose and VITS audio graphs;
- nested `If`/`Loop` and sequence operators;
- multiple ONNX modules coordinated by a host inference loop.

The generic compiler should reach these features through reusable contracts.
The package-specific orchestration layer may provide tokenization, G2P, audio
resampling, and host scheduling, but it must call compiled Vulkan subprograms
through the same package/runtime interfaces as other models.

## Delivery phases

### Phase A: contracts and static correctness

- capability-aware kernel registry;
- profile-aware static compilation;
- explicit normalized tensor specs;
- complete diagnostics and no partial packages;
- FP32 elementwise, views, and layout test coverage.

### Phase B: useful neural networks

- MatMul/Gemm, Conv, reductions, Softmax, normalization;
- layout propagation and bounded Shape profiles;
- tensor lifetime planning and package runtime profiles;
- static CNN and Transformer block end-to-end tests.

### Phase C: model control and quantization

- host-driven subprogram calls;
- KV-cache/state tensors;
- INT8 dequantization and integer accumulation;
- offline INT4 weight dequantization;
- sequence/control-flow normalization.

### Phase D: portable Vulkan distribution

- feature and limit negotiation;
- device compatibility certificates;
- native FP16/INT8/INT4 kernels;
- asynchronous execution and allocation reuse;
- package compatibility and migration policy.

## Non-negotiable invariants

- No silent CPU fallback inside a Vulkan package.
- No package is written after a compilation error.
- Every runtime tensor has a declared dtype, shape/profile, and layout.
- Every shader is associated with a descriptor contract and capability range.
- Every verified result records the reference runtime, device, profile, and tolerance.
- Every unsupported operation has a source node and actionable diagnostic.
