# Architecture

## Product boundary

Torch to Vulcan is an offline ONNX graph editor and compiler with a separate
Vulkan execution runtime. The editor is not an unrestricted diagramming tool:
every edit must preserve tensor types, graph topology, and operator contracts.

```text
ONNX model
    |
    v
Python importer -----> Graph IR <-----> React Web UI
                           |
                           v
                    TTV-Expr semantic IR
                           |
                           v
                    schedule/kernel IR
                           |
              +------------+------------+
              |                         |
              v                         v
       Torch reference             kernel registry
                                        |
                                        v
                                  GLSL -> SPIR-V
                                        |
                                        v
                               compiled model package
                                        |
                                        v
                                 Vulkan runtime
```

Torch is a reference backend for numerical comparison. It is deliberately not
an intermediate representation and is not part of the production runtime.

The long-term ONNX and Vulkan coverage contract is defined in
[`onnx-vulkan-coverage.md`](onnx-vulkan-coverage.md). It separates semantic
support, normalization, backend lowering, SPIR-V compilation, and device
verification instead of treating them as one boolean operator flag.

## Components

### Web UI

The Web UI uses React, TypeScript, and XYFlow. It has three primary
regions: model/operator navigation, the graph canvas, and a selected-node
inspector. Connections are derived from tensor producer and consumer
relationships; the Graph IR does not store a second, redundant edge list.

The inspection milestone communicates through a typed JSON report produced by
FastAPI. Graph editing will communicate through versioned Graph IR JSON. The
Web UI must not depend on Python object serialization or raw ONNX protobuf
messages.

#### Graph layout

Canvas layout is derived from the complete tensor producer/consumer graph and
does not rely on ONNX node array order. The layout pipeline is:

1. resolve all tensor producers and consumer edges;
2. find strongly connected components with Tarjan's algorithm;
3. collapse components into a DAG and assign longest-path ranks;
4. pin graph inputs to the left rank and graph outputs to one rightmost rank;
5. move root `Constant` nodes next to their earliest consumers;
6. run alternating barycentric sweeps to reduce crossings within each rank;
7. expand cyclic components vertically and render their back edges distinctly.

ONNX `Loop` and `Scan` nodes are marked as control-flow loops even when the
surrounding ONNX graph is acyclic. Every input and output port has a separate
XYFlow handle so parallel tensor edges do not share a single endpoint.
The operator navigator groups identical `(domain, op_type)` pairs while retaining
the first concrete node as its selection target. Tensor types and shapes are read
from the inspection report after best-effort ONNX shape inference; missing
intermediate metadata is rendered as `UNKNOWN` instead of being inferred in the
browser. Tensor reports carry `shape_known` so an unknown rank cannot be confused
with a zero-dimensional scalar.

#### Hierarchical preview

Large reports open at a model-pipeline overview instead of immediately rendering
every ONNX operator. Each ONNX file is a module whose complete input/output
signature remains available in the inspector. Exact cross-model tensor-name
matches are rendered as confirmed connections. Matches based on conventions such
as `present_*` to `past_*` are amber dashed candidates because an ONNX archive does
not contain an authoritative cross-file execution manifest.

Double-clicking a module descends through ONNX node-name scopes. Boundary tensors
between scopes are aggregated into one bus per source/target pair. A leaf scope
restores exact operator and tensor edges; unscoped regions above the leaf budget
are split into topology ranges of at most 80 operators. Breadcrumbs preserve the
current location and provide direct navigation back to the pipeline.

All regular connections use Bezier curves between their concrete source and
target tensor handles. Graph inputs use green nodes and edges, graph outputs use
amber, and cycle or inferred edges retain distinct dashed styling.

Boundary-state autoregression is represented as a Blender-style loop zone. A
model is considered a loop candidate when its outputs feed its own inputs through
strong state conventions such as `present_*` to `past_*`, identical state names,
or `y` to `iy`. This self-feedback is recorded in addition to any first-iteration
initializer edge from another model. The zone displays the feedback tensor count,
uses a dedicated edge routed from the body output handle back to its input handle,
and labels `stop`, `condition`, `finished`, or `eos` outputs as exit conditions.
Because this relationship is reconstructed from signatures rather than an
execution manifest, the loop remains visually marked as inferred.

### Compiler service

The first compiler implementation is Python so it can directly use the ONNX
and Torch ecosystems. Its responsibilities are:

1. import and normalize ONNX models;
2. infer tensor shape and type information;
3. validate graph edits;
4. run graph optimization passes;
5. select registered Vulkan kernels;
6. generate and validate SPIR-V;
7. compare Vulkan output with the Torch reference backend;
8. emit a compiled model package.

The HTTP API will be a thin adapter around compiler application services. Core
passes must remain callable without starting a server.

### Operator semantic language

`TTV-Expr 0.1` is the structured, hardware-independent mathematical layer
between normalized Graph IR and kernel scheduling. Each supported ONNX operator
is lowered into a structured expression AST containing tensor accesses,
assignments, constraints, parallel iteration, and metadata-only views. Compiler
passes consume this AST; formatted pseudocode is never parsed back into the
compiler.

Semantic resolution receives a `NodeContext` containing the operator domain,
opset, attributes, input/output tensor specifications, overload, and small
constant inputs. Resolution order is:

1. expand a matching model-local `FunctionProto`;
2. apply an audited versioned registry rule;
3. expand a static function body published by the ONNX operator schema;
4. report the operator as unknown.

Function bodies lower to structured `Invoke` statements rather than formatted
strings, so later compiler passes can recursively lower the decomposition.
Context-dependent schema functions are deferred until their type/shape API can
be supplied completely. The importer may bind scalar or small tensor constants
of at most 16 elements, such as axes and target sizes. It never reads large
weights into semantic reports.

The registry is version-aware and chooses a definition from the ONNX domain,
operator type, opset, and semantically relevant attributes. A model report stores
each resulting definition once under a stable semantic key, while operator
instances only reference that key. This avoids repeating the same explanation
for graphs containing thousands of identical operators. Tensor-valued ONNX
attributes are represented by type and shape summaries rather than embedded
payloads.

Every expression program has English and Chinese pretty-printers over the same
AST. The Web UI defaults to Chinese and can switch languages without reimporting
the model; the selection remains active while navigating between nodes. Stable
symbols such as tensor names and indices are intentionally unchanged between
languages so the two renderings can be compared line by line. An undefined
operator is reported explicitly and must not be mistaken for a compilable
kernel.

Each definition reports its provenance and confidence. `EXACT_FUNCTION` means
the expression came from an ONNX function body, `EXACT_RULE` means it came from
an audited registry rule, and `UNKNOWN` means no semantic lowering exists.
Reference-runtime differential verification is recorded as a separate compiler
certificate; neither exact source category claims that a Vulkan kernel has passed
numerical validation.

The foundational semantic catalog covers elementwise arithmetic and comparisons,
shape and view operations, indexing, layout transforms, reductions, type
conversion, quantization, normalization, convolution, and linear algebra. `Cast`
records the target tensor type and version-specific float8 saturation and rounding
controls. `Reshape` is classified as a view that normally emits no compute shader,
while `Transpose` records a layout transformation.

Control flow, sequence manipulation, signal transforms, and vendor-specific
operators require dedicated semantic designs and remain explicitly undefined until
those designs exist. This semantic support does not by itself mean that a Vulkan
kernel exists; scheduling, validation, and reference comparison remain separate
later stages.

### Kernel registry

Each normalized operator owns separate contracts for shape inference, Torch
reference behavior, and Vulkan kernel candidates. A candidate declares its
supported data types, layouts, device capabilities, specialization constants,
and cost estimate. This keeps PyTorch behavior out of shader generation.

Vulkan lowering returns a `DispatchPlan`, not necessarily one shader. A plan may
contain zero dispatches for metadata-only views, one dispatch for simple
elementwise kernels, or multiple dispatches and barriers for reductions and
other staged algorithms. Mapping verification is incremental and auditable; see
[`vulkan-verification.md`](vulkan-verification.md).

Verified plans become durable only after package materialization. The executable
package contract, artifact hashing, constant layout, and staged implementation are
defined in [`executable-package.md`](executable-package.md).

### Vulkan runtime

The Python verification runtime can allocate host-visible buffers or opt into
device-local tensor buffers with persistent staging allocations. It creates
descriptor sets and compute pipelines, records linear dispatch/copy command
variants, inserts transfer/compute/host barriers, waits on a fence, and compares
GPU output with ONNX Runtime. The device-local path establishes the first real
GPU execution baseline, but readback and queue synchronization still dominate
small or I/O-heavy graphs; tensor arenas, lifetime reuse, fusion, and async
double buffering remain production-runtime work. Zig remains the preferred
language candidate for the packaged runtime after the package and execution
plan stabilize.

## Dependency direction

Dependencies always point inward toward stable contracts:

```text
Web UI ---------> Graph IR <--------- ONNX adapter
                      ^
                      |
compiler passes ------+------ kernel implementations
                      |
package writer -------+------ Torch reference adapter
```

Graph IR cannot import Web UI, ONNX, Torch, Vulkan, or transport-specific
types. Backend annotations created during compilation live in compiler state or
the compiled manifest, not in the semantic Graph IR.

## Initial non-goals

- training or automatic differentiation;
- arbitrary ONNX control flow;
- executing ONNX directly in the Vulkan runtime;
- embedding weights as JSON arrays;
- supporting dynamic shapes without declared bounds;
- optimizing compiler latency before profiling identifies a bottleneck.
