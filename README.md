# Torch to Vulcan

Torch to Vulcan is a visual ONNX-to-Vulkan compute compiler. The project will
provide a node-based Web UI for inspecting and rewriting ONNX graphs, an
offline compiler that lowers graph operators to compute shaders, and a small
Vulkan runtime.

> The project name uses "Vulcan"; the graphics and compute API targeted by the
> compiler is Vulkan.

## Current scope

This repository currently provides the first importer and defines the contracts
on which the Web UI, compiler, and runtime will be built:

- direct ONNX and ZIP-contained ONNX operator inspection;
- a versioned Graph IR JSON Schema;
- an initial compiled-model manifest schema;
- dependency-free Python Graph IR data structures and validation;
- an example graph and unit tests;
- architecture, development, and format documentation.

## Repository layout

```text
docs/                   Architecture and contributor documentation
examples/               Small, reviewable Graph IR examples
schemas/                Language-neutral JSON contracts
src/torch_to_vulcan/    Python compiler package
tests/                  Compiler and contract tests
```

Planned components will live in `web/`, `compiler/`, and `runtime/` once their
toolchains are introduced. The Python package under `src/` is the initial
compiler core and ONNX adapter home.

## Quick start

Python 3.11 or later is required.

```powershell
py -3.11 -m pip install -e ".[dev]"
py -3.11 -m unittest discover -s tests -v
py -3.11 -m torch_to_vulcan validate examples/relu.graph.json
ttv inspect path\to\models.zip
ttv inspect path\to\model.onnx
```

See [docs/development.md](docs/development.md) for the development workflow and
[docs/architecture.md](docs/architecture.md) for subsystem boundaries. The
current ZIP importer is documented in [docs/importer.md](docs/importer.md).
