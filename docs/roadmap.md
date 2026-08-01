# Roadmap

## Milestone 0: Contracts

- versioned Graph IR and package manifest;
- Python data model and invariant validation;
- architecture and contribution documentation;
- small graph fixtures and tests.

## Milestone 1: ONNX inspection

- direct/compressed ONNX and common archive-based recursive operator inspection;
- ONNX-to-Graph-IR importer with deterministic IDs;
- shape/type inference and initializer extraction;
- FastAPI graph endpoints;
- React three-panel shell and XYFlow graph rendering;
- operator support diagnostics;
- read-only import and Graph IR export.

## Milestone 2: Graph editing

- attribute editing with operator-aware forms;
- validated edge reconnection;
- insertion, replacement, deletion, and bypass operations;
- undo/redo and persisted UI positions;
- constant folding and canonicalization passes.

## Milestone 3: First shaders

- kernel registry and GLSL template infrastructure;
- backend capability declarations and profile-aware compilation contracts;
- SPIR-V compilation and validation;
- FP32 buffer kernels for Add, Mul, Relu, Reshape, and Transpose;
- Torch reference execution and differential tests;
- shader and dispatch-plan inspection in the Web UI;
- static root-graph compilation to an integrity-checked TTV 0.1 package.

## Milestone 4: Useful neural networks

- MatMul/Gemm (二维 FP32 baseline is now implemented), Conv2D, DepthwiseConv,
  pooling, and Softmax;
- explicit layout propagation and conversion;
- tensor lifetime analysis and memory reuse;
- Vulkan runtime prototype;
- end-to-end execution of a small static-shape CNN.

## Later work

- normalized ONNX IR with symbolic dimensions, layouts, subprograms, and
  quantization metadata (the first normalized tensor/node/graph IR is now in
  place; subprogram and quantization records remain);
- FP16 and capability-based kernel selection;
- operator fusion and autotuning;
- bounded dynamic shapes;
- Transformer operators;
- INT8 quantization;
- stable compiled-package distribution and compatibility policy.
