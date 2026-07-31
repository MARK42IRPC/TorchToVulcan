# Torch to Vulcan

[中文](#中文) | [English](#english)

> 项目名使用 **Vulcan**；编译器实际面向的图形与计算 API 是 **Vulkan**。

## 中文

Torch to Vulcan 是一个可视化 ONNX-to-Vulkan 计算编译器。项目由节点式 WebUI、离线图编译器和轻量 Vulkan Runtime 组成，目标是将 ONNX 前向推理图转换为可在通用 GPU 上运行的计算着色器。

### 当前能力

- 直接导入 `.onnx`，以及 `.onnx.gz/.bz2/.xz` 单文件压缩模型；
- 导入 ZIP、7z、TAR、TGZ、TBZ2、TXZ 等归档并递归发现其中的 ONNX 模型；
- 识别主图与嵌套子图中的算子；
- 在 WebUI 中浏览模型、算子、输入输出节点和 Tensor 连线；
- 输出适合 WebUI 或其他工具消费的 JSON 检查报告；
- 版本化 Graph IR 与编译模型包 Schema；
- Python Graph IR 数据结构、跨对象校验和测试。

`ckpt`、`pth` 和其他 Torch 容器格式计划通过独立的安全适配器接入，目前尚未实现。
RAR 目前也未启用，因为跨平台解析需要额外的 `unrar` 或 `bsdtar` 程序。

### 技术栈

- WebUI：React、TypeScript、Vite、XYFlow；
- 导入与 API：Python、ONNX、FastAPI；
- 参考计算：Torch，后续接入；
- Shader：GLSL Compute 到 SPIR-V，后续接入；
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
.venv\Scripts\ttv inspect path\to\models.zip --summary-only
.venv\Scripts\ttv inspect path\to\models.zip --json
```

### 测试与构建

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
npm --prefix web run build
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
- import ZIP, 7z, TAR, TGZ, TBZ2, and TXZ archives and recursively discover ONNX models;
- inspect operators in root graphs and nested subgraphs;
- browse models, operators, graph inputs/outputs, and tensor connections in the Web UI;
- emit machine-readable JSON inspection reports;
- maintain versioned Graph IR and compiled-package schemas;
- validate compiler-side Graph IR structures and cross-object invariants.

`ckpt`, `pth`, and other Torch container formats are planned as separate security-aware source adapters and are not implemented yet.
RAR is also deferred because cross-platform parsing requires an external `unrar` or `bsdtar` executable.

### Technology stack

- Web UI: React, TypeScript, Vite, and XYFlow;
- importer and API: Python, ONNX, and FastAPI;
- reference execution: Torch, planned;
- shaders: GLSL Compute to SPIR-V, planned;
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
.venv\Scripts\ttv inspect path\to\models.zip --summary-only
.venv\Scripts\ttv inspect path\to\models.zip --json
```

### Test and build

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
npm --prefix web run build
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
