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
                    compiler passes
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
from the inspection report; missing intermediate metadata is rendered as
`UNKNOWN` instead of being inferred in the browser.

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

### Kernel registry

Each normalized operator owns separate contracts for shape inference, Torch
reference behavior, and Vulkan kernel candidates. A candidate declares its
supported data types, layouts, device capabilities, specialization constants,
and cost estimate. This keeps PyTorch behavior out of shader generation.

### Vulkan runtime

The runtime only loads a compiled package, allocates resources, creates cached
pipelines, and submits the recorded dispatch plan. Zig is the preferred
language candidate for this component; the choice is deferred until the model
package and execution plan stabilize.

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
