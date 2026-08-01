# Torch to Vulcan

[中文](#中文) | [English](#english)

> 项目名使用 **Vulcan**；编译器实际面向的图形与计算 API 是 **Vulkan**。

## 中文

Torch to Vulcan 是一个可视化 ONNX-to-Vulkan 计算编译器。项目由节点式 WebUI、离线图编译器和轻量 Vulkan Runtime 组成，目标是将 ONNX 前向推理图转换为可在通用 GPU 上运行的计算着色器。

### 当前能力

- 直接导入 `.onnx`，以及 `.onnx.gz/.bz2/.xz` 单文件压缩模型；
- 导入 ZIP、RAR、7z、TAR、TGZ、TBZ2、TXZ 等归档并递归发现其中的 ONNX 模型；
- 识别主图与嵌套子图中的算子；
- 在 WebUI 中浏览模型、算子、输入输出节点和 Tensor 连线；
- 导入时显示归档扫描、解压和逐模型解析进度；
- 当模型估算大小超过当前可用内存的 60% 时，要求用户明确确认后继续；
- 输出适合 WebUI 或其他工具消费的 JSON 检查报告；
- 版本化 Graph IR 与编译模型包 Schema；
- Python Graph IR 数据结构、跨对象校验和测试；
- 将支持的静态 ONNX 根图编译为带完整性校验的 TTV 0.1 可执行目录包；
- 提供首个规范化 ONNX IR，保留符号维度、layout、stride、常量和嵌套子图；
- 将二维 FP32 `MatMul`/`Gemm` 编译为 Vulkan compute shader，并可在真实 Vulkan
  设备上执行和与 ONNX Runtime 做差分验证；
- 对常量/shape 子图做编译期求值，将 `Identity`/`Reshape` 落为视图，并将已注册的 FP32 kernel 编译为 SPIR-V。
- 通过 Vulkan 运行时执行已落盘的线性 TTV 程序，支持持久设备、pipeline、descriptor 和 command buffer。

`ckpt`、`pth` 和其他 Torch 容器格式计划通过独立的安全适配器接入，目前尚未实现。
Windows x64 版内置 UnRAR 解码器，可开箱导入 RAR；其他平台可通过
`TTV_UNRAR` 指定 `unrar` 路径，或将其安装到 `PATH`。

### 技术栈

- WebUI：React、TypeScript、Vite、XYFlow；
- 导入与 API：Python、ONNX、FastAPI；
- 参考计算：ONNX Runtime，Torch CPU 作为后续的第二参考后端；
- Shader：GLSL Compute 到 SPIR-V，已接入便携 glslang 与 Vulkan 差分验证；
- Runtime：Vulkan，候选实现语言为 Zig。

### 安装

需要 Python 3.11+、Node.js 20+ 和 Git。Windows：

```powershell
.\scripts\setup.ps1
```

也可以直接双击根目录的 `install-deps.bat`。

Linux/macOS：

```bash
./scripts/setup.sh
```

脚本会创建 `.venv`，安装 Python 的开发与 Web 依赖，并安装锁定的前端依赖。

### 启动 WebUI

Windows：

```powershell
.\scripts\dev.ps1
```

也可以直接双击根目录的 `start.bat`。

Linux/macOS：

```bash
./scripts/dev.sh
```

WebUI 默认运行在 `http://127.0.0.1:5173`，API 默认运行在 `http://127.0.0.1:8000`。

### 命令行

```powershell
.venv\Scripts\ttv inspect path\to\model.onnx
.venv\Scripts\ttv inspect path\to\models.zip
.venv\Scripts\ttv inspect path\to\models.tar.gz
.venv\Scripts\ttv inspect path\to\models.7z
.venv\Scripts\ttv inspect path\to\models.rar
.venv\Scripts\ttv inspect path\to\models.zip --summary-only
.venv\Scripts\ttv inspect path\to\models.zip --json
.venv\Scripts\ttv compile path\to\model.onnx artifacts\model.ttv
.venv\Scripts\ttv compile path\to\dynamic-model.onnx artifacts\profiled.ttv --shape-profile examples\shape-profile.json
.venv\Scripts\ttv validate-package artifacts\model.ttv
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --output outputs.npz
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --warmup 3 --iterations 20
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --device-local
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --device-local --resident
```

`compile` 首版仅接受完全静态的根图和已注册 kernel；动态 shape、控制流与未支持算子会输出节点定位诊断，不会留下半成品包。
如果模型的符号维度可以在编译时确定，可以通过 `--shape-profile` 提供一个 JSON
profile，将该模型特化为当前 TTV 0.1 静态包；这不是运行时动态 shape 支持。
`inputs.npz` 的 key 必须与 package manifest 中的输入 tensor ID 一致。runtime 默认使用
host-visible buffer；`--device-local` 会使用设备局部 tensor 和 staging 拷贝，若设备不支持则回退并打印实际模式。
`--resident` 只测输入输出已驻留 GPU 时的稳态调度。大模型的统一显存 arena、生命周期复用、算子融合和异步流水仍在迭代中。
当前 `MatMul`/`Gemm` 只覆盖二维 FP32；批量矩阵、FP16/INT8/INT4、运行时动态 shape
和控制流仍会明确阻止生成 TTV 0.1 包，详见
[`ONNX/Vulkan 覆盖路线`](docs/onnx-vulkan-coverage.md)。

### 测试与构建

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
npm --prefix web test
npm --prefix web run build
npm --prefix web run test:e2e
```

### 仓库结构

```text
docs/                   架构、格式与开发文档
examples/               小型 Graph IR 示例
schemas/                跨语言 JSON 契约
scripts/                安装与开发启动脚本
src/torch_to_vulcan/    导入器、Graph IR 和 HTTP API
tests/                  Python 单元与契约测试
web/                    React 节点式 WebUI
```

进一步阅读：[架构](docs/architecture.md)、[导入器](docs/importer.md)、[Graph IR](docs/graph-ir.md)、[开发流程](docs/development.md)和[路线图](docs/roadmap.md)。

---

## English

Torch to Vulcan is a visual ONNX-to-Vulkan compute compiler. It combines a node-based Web UI, an offline graph compiler, and a small Vulkan runtime with the goal of lowering ONNX inference graphs into compute shaders for general-purpose GPUs.

### Current capabilities

- import direct `.onnx` files and `.onnx.gz/.bz2/.xz` compressed models;
- import ZIP, RAR, 7z, TAR, TGZ, TBZ2, and TXZ archives and recursively discover ONNX models;
- inspect operators in root graphs and nested subgraphs;
- browse models, operators, graph inputs/outputs, and tensor connections in the Web UI;
- show archive scanning, extraction, and per-model parsing progress during import;
- request explicit confirmation when a model estimate exceeds 60% of currently available memory;
- emit machine-readable JSON inspection reports;
- maintain versioned Graph IR and compiled-package schemas;
- validate compiler-side Graph IR structures and cross-object invariants;
- compile supported static ONNX root graphs into integrity-checked TTV 0.1 executable directory packages;
- normalize ONNX models into an internal IR that preserves symbolic dimensions, layouts, strides, constants, and nested graphs;
- lower two-dimensional FP32 `MatMul`/`Gemm` into Vulkan compute shaders, execute them on a real Vulkan device, and compare them with ONNX Runtime;
- evaluate constant/shape subgraphs at compile time, lower `Identity`/`Reshape` to views, and compile registered FP32 kernels to SPIR-V.
- execute a materialized linear TTV program through the Vulkan runtime and benchmark steady-state latency;

`ckpt`, `pth`, and other Torch container formats are planned as separate security-aware source adapters and are not implemented yet.
An UnRAR decoder is bundled for Windows x64, so RAR imports work out of the
box. On other platforms, set `TTV_UNRAR` or install `unrar` on `PATH`.

### Technology stack

- Web UI: React, TypeScript, Vite, and XYFlow;
- importer and API: Python, ONNX, and FastAPI;
- reference execution: ONNX Runtime, with Torch CPU planned as a second reference backend;
- shaders: GLSL Compute to SPIR-V through portable glslang and Vulkan differential verification;
- runtime: Vulkan, with Zig as the current implementation candidate.

### Setup

Python 3.11+, Node.js 20+, and Git are required. On Windows:

```powershell
.\scripts\setup.ps1
```

You can also double-click `install-deps.bat` in the repository root.

On Linux or macOS:

```bash
./scripts/setup.sh
```

The script creates `.venv`, installs Python development and Web dependencies, and installs the locked frontend dependencies.

### Start the Web UI

On Windows:

```powershell
.\scripts\dev.ps1
```

You can also double-click `start.bat` in the repository root.

On Linux or macOS:

```bash
./scripts/dev.sh
```

The Web UI defaults to `http://127.0.0.1:5173`; the API defaults to `http://127.0.0.1:8000`.

### Command line

```powershell
.venv\Scripts\ttv inspect path\to\model.onnx
.venv\Scripts\ttv inspect path\to\models.zip
.venv\Scripts\ttv inspect path\to\models.tar.gz
.venv\Scripts\ttv inspect path\to\models.7z
.venv\Scripts\ttv inspect path\to\models.rar
.venv\Scripts\ttv inspect path\to\models.zip --summary-only
.venv\Scripts\ttv inspect path\to\models.zip --json
.venv\Scripts\ttv compile path\to\model.onnx artifacts\model.ttv
.venv\Scripts\ttv compile path\to\dynamic-model.onnx artifacts\profiled.ttv --shape-profile examples\shape-profile.json
.venv\Scripts\ttv validate-package artifacts\model.ttv
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --output outputs.npz
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --warmup 3 --iterations 20
.venv\Scripts\ttv run artifacts\model.ttv inputs.npz --device-local
.venv\Scripts\ttv benchmark artifacts\model.ttv inputs.npz --device-local --resident
```

The first `compile` implementation accepts only fully static root graphs whose kernels are registered. Dynamic shapes, control flow, and unsupported operators produce node-local diagnostics without leaving a partial package.
When symbolic dimensions are known at compile time, `--shape-profile` specializes
the model into a TTV 0.1 static package. This is compile-time specialization,
not runtime dynamic-shape execution.
The `.npz` input keys must match manifest tensor IDs. The runtime defaults to
host-visible storage for portability. `--device-local` opts into device-local
tensor buffers with persistent host staging copies; if the selected Vulkan
device cannot provide a suitable device-local memory type, execution falls back
to host-visible storage and reports the actual mode. `--resident` measures the
steady-state dispatch command buffer without input upload or output readback.
Small graphs can still lose to CPU because queue submission and synchronization
dominate their latency; use a representative tensor size and report both
end-to-end and resident timings before claiming a speedup.
The current `MatMul`/`Gemm` baseline is two-dimensional FP32 only; batched
matrices, FP16/INT8/INT4, runtime dynamic shapes, and control flow still block
TTV 0.1 package generation. See the
[`ONNX/Vulkan coverage roadmap`](docs/onnx-vulkan-coverage.md).

### Test and build

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
npm --prefix web test
npm --prefix web run build
npm --prefix web run test:e2e
```

### Repository layout

```text
docs/                   Architecture, format, and development documentation
examples/               Small Graph IR examples
schemas/                Language-neutral JSON contracts
scripts/                Setup and development launch scripts
src/torch_to_vulcan/    Importer, Graph IR, and HTTP API
tests/                  Python unit and contract tests
web/                    React node-based Web UI
```

Read more in the [architecture](docs/architecture.md), [importer](docs/importer.md), [Graph IR](docs/graph-ir.md), [development](docs/development.md), and [roadmap](docs/roadmap.md) documents.
