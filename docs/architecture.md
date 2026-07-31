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

The planned Web UI uses React, TypeScript, and XYFlow. It has three primary
regions: model/operator navigation, the graph canvas, and a selected-node
inspector. Connections are derived from tensor producer and consumer
relationships; the Graph IR does not store a second, redundant edge list.

The Web UI communicates through versioned Graph IR JSON and must not depend on
Python object serialization or raw ONNX protobuf messages.

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

