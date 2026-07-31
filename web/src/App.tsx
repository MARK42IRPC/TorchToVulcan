import { useCallback, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import {
  AlertTriangle,
  Box,
  ChevronRight,
  Cpu,
  FileArchive,
  FileUp,
  LoaderCircle,
  Search,
  Upload,
  X,
} from "lucide-react";

import { inspectModel } from "./api";
import { GraphCanvas } from "./GraphCanvas";
import type {
  CanvasNodeData,
  GraphReport,
  InspectionReport,
  ModelReport,
  OperatorReport,
} from "./types";

function displayDomain(domain: string): string {
  return domain || "ai.onnx";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [modelIndex, setModelIndex] = useState(0);
  const [graphPath, setGraphPath] = useState("");
  const [selectedNode, setSelectedNode] = useState<CanvasNodeData | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const model = report?.models[modelIndex] ?? null;
  const graph = model?.graphs.find((item) => item.path === graphPath) ?? model?.graphs[0] ?? null;

  const operators = useMemo(
    () => model?.graphs.flatMap((item) => item.operators) ?? [],
    [model],
  );
  const filteredOperators = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return operators;
    return operators.filter((operator) =>
      [operator.op_type, operator.name, operator.domain, operator.graph_path]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [operators, query]);

  const loadFile = useCallback(async (file: File) => {
    setSelectedFile(file);
    setLoading(true);
    setError("");
    setSelectedNode(null);
    try {
      const nextReport = await inspectModel(file);
      setReport(nextReport);
      setModelIndex(0);
      setGraphPath(nextReport.models[0]?.graphs[0]?.path ?? "");
      if (nextReport.models.length === 0) {
        setError(nextReport.errors[0]?.message ?? "未发现可读取的 ONNX 模型");
      }
    } catch (reason) {
      setReport(null);
      setError(reason instanceof Error ? reason.message : "模型导入失败");
    } finally {
      setLoading(false);
    }
  }, []);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void loadFile(file);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void loadFile(file);
  };

  const selectModel = (index: number) => {
    setModelIndex(index);
    setGraphPath(report?.models[index]?.graphs[0]?.path ?? "");
    setSelectedNode(null);
  };

  const selectOperator = (operator: OperatorReport) => {
    setGraphPath(operator.graph_path);
    setSelectedNode({
      kind: "operator",
      label: operator.op_type,
      subtitle: operator.name || `${displayDomain(operator.domain)} / ${operator.index}`,
      inputs: operator.inputs,
      outputs: operator.outputs,
      operator,
    });
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <strong>TORCH TO VULCAN</strong>
            <span>GRAPH INSPECTOR / 0.1</span>
          </div>
        </div>
        <div className="topbar__signal">
          <span className="signal-dot" />
          IMPORT SERVICE
        </div>
        <button
          type="button"
          className="icon-command"
          title="导入模型"
          aria-label="导入模型"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={17} />
        </button>
      </header>

      <aside className="sidebar">
        <section className="source-section">
          <div className="section-heading">
            <span>SOURCE</span>
            <b>01</b>
          </div>
          <div
            className={`upload-tool${dragging ? " is-dragging" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <FileUp size={20} strokeWidth={1.6} />
            <div className="upload-tool__copy">
              <strong>{selectedFile?.name ?? "导入模型"}</strong>
              <span>
                {selectedFile ? `${formatBytes(selectedFile.size)} / ${report?.source_type ?? "..."}` : "ONNX / ARCHIVE"}
              </span>
            </div>
            <button type="button" onClick={() => fileInputRef.current?.click()}>
              选择文件
            </button>
          </div>
          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept=".onnx,.onnx.gz,.onnx.bz2,.onnx.xz,.zip,.tar,.tar.gz,.tgz,.tar.bz2,.tbz2,.tar.xz,.txz,.7z,application/zip,application/gzip,application/x-7z-compressed,application/x-tar,application/octet-stream"
            onChange={onFileChange}
          />
          {error && (
            <div className="inline-error" role="alert">
              <AlertTriangle size={14} />
              <span>{error}</span>
              <button type="button" title="关闭" aria-label="关闭" onClick={() => setError("")}>
                <X size={13} />
              </button>
            </div>
          )}
        </section>

        <section className="model-section">
          <div className="section-heading">
            <span>MODELS</span>
            <b>{String(report?.models.length ?? 0).padStart(2, "0")}</b>
          </div>
          <div className="model-list">
            {report?.models.map((item, index) => (
              <button
                type="button"
                className={index === modelIndex ? "is-active" : ""}
                key={`${item.path}-${index}`}
                onClick={() => selectModel(index)}
              >
                {report.source_type !== "onnx" ? <FileArchive size={14} /> : <Cpu size={14} />}
                <span>{item.path}</span>
                <b>{item.operator_count}</b>
              </button>
            ))}
            {!report && <span className="list-empty">NO MODEL</span>}
          </div>
        </section>

        <section className="operator-section">
          <div className="section-heading">
            <span>OPERATORS</span>
            <b>{String(operators.length).padStart(3, "0")}</b>
          </div>
          <label className="search-box">
            <Search size={14} aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索算子"
              aria-label="搜索算子"
            />
          </label>
          <div className="operator-list">
            {filteredOperators.map((operator) => (
              <button
                type="button"
                key={`${operator.graph_path}-${operator.index}`}
                className={selectedNode?.operator === operator ? "is-active" : ""}
                onClick={() => selectOperator(operator)}
              >
                <Box size={14} strokeWidth={1.7} />
                <span>
                  <strong>{operator.op_type}</strong>
                  <small>{operator.name || `${displayDomain(operator.domain)} / ${operator.index}`}</small>
                </span>
                <ChevronRight size={13} />
              </button>
            ))}
            {model && filteredOperators.length === 0 && <span className="list-empty">NO MATCH</span>}
          </div>
        </section>
      </aside>

      <section className="workspace">
        <div className="workspace-bar">
          <div>
            <span>COMPUTE GRAPH</span>
            <strong>{graph?.path ?? "UNLOADED"}</strong>
          </div>
          {model && (
            <label className="graph-select">
              <span>GRAPH</span>
              <select
                value={graph?.path ?? ""}
                onChange={(event) => {
                  setGraphPath(event.target.value);
                  setSelectedNode(null);
                }}
              >
                {model.graphs.map((item) => (
                  <option key={item.path} value={item.path}>
                    {item.path}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <div className="canvas-frame">
          <GraphCanvas graph={graph} onSelectNode={setSelectedNode} />
          {loading && (
            <div className="loading-screen" role="status">
              <LoaderCircle size={22} className="spin" />
              <strong>ANALYZING GRAPH</strong>
            </div>
          )}
        </div>
      </section>

      <aside className="inspector">
        <div className="section-heading">
          <span>INSPECTOR</span>
          <b>03</b>
        </div>
        <NodeInspector node={selectedNode} model={model} graph={graph} />
      </aside>

      <footer className="statusbar">
        <span className={report ? "status-ready" : ""}>{report ? "READY" : "IDLE"}</span>
        <span>MODELS {report?.models.length ?? 0}</span>
        <span>OPS {report?.operator_count ?? 0}</span>
        <span>ERRORS {report?.errors.length ?? 0}</span>
        <b>VULKAN TARGET / OFFLINE</b>
      </footer>
    </main>
  );
}

function NodeInspector({
  node,
  model,
  graph,
}: {
  node: CanvasNodeData | null;
  model: ModelReport | null;
  graph: GraphReport | null;
}) {
  if (!node) {
    return (
      <div className="inspector-empty">
        <Cpu size={24} strokeWidth={1.4} />
        <strong>{model?.graph_name ?? "NO SELECTION"}</strong>
        <dl>
          <div><dt>IR</dt><dd>{model?.ir_version ?? "--"}</dd></div>
          <div><dt>GRAPH</dt><dd>{graph?.operators.length ?? 0} OPS</dd></div>
          <div><dt>OPSET</dt><dd>{model?.opsets.map((item) => item.version).join(", ") || "--"}</dd></div>
        </dl>
      </div>
    );
  }

  return (
    <div className="node-detail">
      <div className="node-detail__title">
        <span>{node.kind.toUpperCase()}</span>
        <strong>{node.label}</strong>
        <small>{node.subtitle}</small>
      </div>
      {node.operator && (
        <dl className="property-list">
          <div><dt>DOMAIN</dt><dd>{displayDomain(node.operator.domain)}</dd></div>
          <div><dt>INDEX</dt><dd>{node.operator.index}</dd></div>
          <div><dt>GRAPH</dt><dd title={node.operator.graph_path}>{node.operator.graph_path}</dd></div>
          <div><dt>KERNEL</dt><dd className="pending">PENDING</dd></div>
        </dl>
      )}
      <TensorList title="INPUT TENSORS" tensors={node.inputs} />
      <TensorList title="OUTPUT TENSORS" tensors={node.outputs} />
    </div>
  );
}

function TensorList({ title, tensors }: { title: string; tensors: string[] }) {
  return (
    <section className="tensor-list">
      <h3>{title}<b>{tensors.length}</b></h3>
      {tensors.map((tensor, index) => (
        <div key={`${tensor}-${index}`}>
          <span>{String(index).padStart(2, "0")}</span>
          <code title={tensor}>{tensor || "OPTIONAL"}</code>
        </div>
      ))}
      {tensors.length === 0 && <span className="list-empty">NONE</span>}
    </section>
  );
}
