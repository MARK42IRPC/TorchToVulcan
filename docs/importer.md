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
ttv inspect path\to\models.zip --summary-only
ttv inspect path\to\models.zip --json
```

The normal output contains an aggregate operator count followed by every model,
graph, and node. Direct ONNX and archive inputs return the same report structure.
JSON output is the integration contract for the first Web UI model browser.
Its `source_type` identifies the selected adapter, so clients do not need to
infer it from a path.

The command exits with:

- `0` when every discovered ONNX entry was parsed;
- `1` when the input is unsupported or cannot be opened safely;
- `2` when the report contains one or more invalid ONNX entries.

Valid models are still reported when another entry is malformed.

## Input behavior

- direct `.onnx`, `.onnx.gz`, `.onnx.bz2`, and `.onnx.xz` files are supported;
- ZIP and 7z archives are supported;
- TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2, and TAR.XZ/TXZ archives are supported;
- filenames ending in `.onnx` are matched case-insensitively at any archive depth;
- entries are processed in deterministic path order;
- the archive is read without extracting files;
- initializer external data is not resolved during operator discovery;
- operators in ONNX `GRAPH` and `GRAPHS` attributes are traversed recursively;
- non-ONNX files are ignored;
- encrypted entries are reported as unsupported;
- RAR is not supported because it requires an external `unrar` or `bsdtar`
  decoder on most platforms.

Default resource limits allow 512 models, 256 MiB per model, and 1 GiB total
uncompressed ONNX data. Callers of the Python API can provide stricter
`InspectionLimits`.

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

Future `ckpt` and `pth` support will be added as separate source adapters that
produce this same inspection report. They are intentionally out of scope for
the initial implementation because loading pickled Torch files requires a
separate trust and sandbox policy.
