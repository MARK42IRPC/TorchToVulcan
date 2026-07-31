# ONNX Inspector

The first importer milestone inspects either a direct ONNX inference model or
all ONNX models inside a supported archive and reports their operators. It does
not yet convert ONNX into Graph IR or run shape inference.

## Command line

```powershell
ttv inspect path\to\model.onnx
ttv inspect path\to\model.onnx.gz
ttv inspect path\to\models.zip
ttv inspect path\to\models.tar.gz
ttv inspect path\to\models.7z
ttv inspect path\to\models.rar
ttv inspect path\to\models.zip --summary-only
ttv inspect path\to\models.zip --json
```

The normal output contains an aggregate operator count followed by every model,
graph, and node. Direct ONNX and archive inputs return the same report structure.
JSON output is the integration contract for the first Web UI model browser.
Its `source_type` identifies the selected adapter, so clients do not need to
infer it from a path. Each graph also exposes a `values` collection containing
the tensor element type and shape found in ONNX graph inputs, outputs,
`value_info`, and initializers. Intermediate values omitted by the source model
remain `UNKNOWN`; the inspector does not run ONNX shape inference yet.

The command exits with:

- `0` when every discovered ONNX entry was parsed;
- `1` when the input is unsupported or cannot be opened safely;
- `2` when the report contains one or more invalid ONNX entries.

Valid models are still reported when another entry is malformed.

## Input behavior

- direct `.onnx`, `.onnx.gz`, `.onnx.bz2`, and `.onnx.xz` files are supported;
- ZIP, RAR, and 7z archives are supported;
- TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2, and TAR.XZ/TXZ archives are supported;
- filenames ending in `.onnx` are matched case-insensitively at any archive depth;
- entries are processed in deterministic path order;
- ZIP, TAR, and 7z are read without persistent extraction; RAR uses a private
  temporary directory to avoid repeated solid-archive decompression and removes
  it when the inspection ends;
- initializer external data is not resolved during operator discovery;
- operators in ONNX `GRAPH` and `GRAPHS` attributes are traversed recursively;
- non-ONNX files are ignored;
- encrypted entries are reported as unsupported;
- Windows x64 uses the bundled UnRAR decoder. Other platforms locate `unrar`
  through `TTV_UNRAR` and then `PATH`.

The default hard limit is 512 discovered models. Model byte limits are optional.
Before parsing, the importer compares the largest model that will be resident
with currently available physical memory. If the estimate exceeds 60%, the Web
UI asks the user to choose `取消加载` or `我知道我在做什么`. Callers of the
Python API can override this behavior with `InspectionLimits` and
`confirm_large_model`.

The Web API also exposes `/api/inspect/stream`, an NDJSON stream that reports
the current scan, extraction, and per-model parsing phase before returning the
final inspection report.

## Python API

```python
from torch_to_vulcan.importer import inspect_path

report = inspect_path("models.tar.gz")  # Direct ONNX and other archives use the same API.
for model in report.models:
    for graph in model.graphs:
        for operator in graph.operators:
            print(model.path, graph.path, operator.op_type)
```

The report contains raw ONNX domains. An empty domain means the standard
`ai.onnx` domain and is only expanded for human-readable CLI output.
The Web UI groups the operator navigator by `(domain, op_type)`, displays the
number of occurrences, and selects the first occurrence when a group is opened.
Tensor metadata is shown both as a compact graph-node summary and in full in
the selected-node inspector.

Archive reports are initially visualized as a pipeline of ONNX modules. The UI
can confirm cross-model edges only when output/input names match exactly. Naming
conventions provide lower-confidence candidate edges, displayed distinctly; a
future package manifest will be the authoritative source for cross-model calls.

Future `ckpt` and `pth` support will be added as separate source adapters that
produce this same inspection report. They are intentionally out of scope for
the initial implementation because loading pickled Torch files requires a
separate trust and sandbox policy.
