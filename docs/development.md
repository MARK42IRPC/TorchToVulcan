# Development

## Prerequisites

- Git 2.28 or later;
- Python 3.11 or later;
- Node.js 20 or later;
- PowerShell 7, Bash, or another shell capable of invoking Python;
- a Python virtual environment.

Vulkan mapping verification additionally requires a Vulkan-capable driver. The
setup script installs ONNX Runtime, the Python Vulkan loader binding, and a
portable glslang compiler. A full Vulkan SDK is optional; when present its native
`glslangValidator` and `spirv-val` tools take precedence.

Torch and Zig remain deferred until the component that needs each tool is
introduced.

## Setup

```powershell
.\scripts\setup.ps1
```

Run the tests and validate the example contract:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
npm --prefix web test
npm --prefix web run build
npm --prefix web run test:e2e
.venv\Scripts\python -m torch_to_vulcan validate examples/relu.graph.json
```

On Unix-like systems, run `./scripts/setup.sh` and use `.venv/bin/python`.

Start both development servers with `.\scripts\dev.ps1` on Windows or
`./scripts/dev.sh` on Unix-like systems.

## Working agreements

1. Treat JSON Schema as the cross-language contract and update it before or
   with language-specific types.
2. Add a migration whenever a persisted Graph IR document changes
   incompatibly.
3. Keep Graph IR independent of React, ONNX, Torch, HTTP, and Vulkan types.
4. Put operator semantics in registry definitions, not in route handlers or UI
   components.
5. Accompany each compiler pass with focused unit tests and at least one graph
   fixture.
6. Validate generated SPIR-V and compare kernel output against a reference
   implementation before declaring an operator supported.
7. Put symbolic dimension bindings in `ShapeProfile`; do not smuggle profile
   values through kernel attributes or package metadata alone.

## Branch and commit convention

Use short topic branches such as `feature/onnx-import` or
`fix/transpose-shape`. Commits should describe one coherent behavior change.
Conventional Commits are recommended but not currently enforced:

```text
feat(ir): add symbolic dimensions
fix(import): preserve optional ONNX inputs
docs: describe kernel registration
```

## Adding an operator

An operator is not considered supported until it has:

1. ONNX normalization rules for the supported opset range;
2. shape and type inference;
3. a Torch or independent CPU reference implementation;
4. at least one Vulkan kernel candidate and capability declaration;
5. normal, edge-shape, and invalid-input tests;
6. GPU/reference numerical comparison with documented tolerances.

The UI support indicator must be derived from the compiler registry rather than
maintained as a separate hard-coded list.

The current static compiler accepts a profile with `ttv compile --shape-profile`
and records the selected profile in package metadata. A profile specializes one
concrete shape; it does not authorize runtime shape changes. Runtime profiles,
bounded symbolic expressions, and multi-profile packages belong to the next
package format.

## Generated files

Do not commit imported ONNX models, SPIR-V binaries, compiled model packages, or
large generated weights unless they are intentionally added as small test
fixtures. The default `.gitignore` excludes these artifacts.
