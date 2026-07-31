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

import { inspectModel, MemoryConfirmationError } from "./api";
import { GraphCanvas } from "./GraphCanvas";
import { buildHierarchyView, scopeForOperator } from "./hierarchy";
import type {
  CanvasNodeData,
  GraphReport,
  HierarchyLocation,
  ImportProgress,
  InspectionReport,
  MemoryWarning,
  ModelReport,
  OperatorReport,
  TensorValueReport,
} from "./types";

function displayDomain(domain: string): string {
  return domain || "ai.onnx";
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function operatorGroupKey(operator: OperatorReport): string {
  return `${operator.domain}\0${operator.op_type}`;
}

function tensorValues(graph: GraphReport, names: string[]): TensorValueReport[] {
  const values = new Map((graph.values ?? []).map((value) => [value.name, value]));
  return names.map((name) => values.get(name) ?? { name, data_type: "UNKNOWN", shape: [] });
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [modelIndex, setModelIndex] = useState(0);
  const [graphPath, setGraphPath] = useState("");
  const [location, setLocation] = useState<HierarchyLocation>({ kind: "overview" });
  const [selectedNode, setSelectedNode] = useState<CanvasNodeData | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [memoryWarning, setMemoryWarning] = useState<MemoryWarning | null>(null);
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(null);

  const model = report?.models[modelIndex] ?? null;
  const hierarchyView = useMemo(
    () => buildHierarchyView(report, location),
    [location, report],
  );
  const graph = hierarchyView.graph
    ?? model?.graphs.find((item) => item.path === graphPath)
    ?? model?.graphs[0]
    ?? null;

  const operators = useMemo(
    () => model?.graphs.flatMap((item) => item.operators) ?? [],
    [model],
  );
  const operatorGroups = useMemo(() => {
    const groups = new Map<
      string,
      { key: string; representative: OperatorReport; instances: OperatorReport[] }
    >();
    operators.forEach((operator) => {
      const key = operatorGroupKey(operator);
      const existing = groups.get(key);
      if (existing) existing.instances.push(operator);
      else groups.set(key, { key, representative: operator, instances: [operator] });
    });
    return [...groups.values()];
  }, [operators]);
  const filteredOperatorGroups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return operatorGroups;
    return operatorGroups.filter((group) =>
      group.instances.some((operator) =>
        [operator.op_type, operator.name, operator.domain, operator.graph_path]
          .join(" ")
          .toLowerCase()
          .includes(needle),
      ),
    );
  }, [operatorGroups, query]);

  const loadFile = useCallback(async (file: File, confirmLargeModel = false) => {
    setSelectedFile(file);
    setLoading(true);
    setError("");
    setMemoryWarning(null);
    setImportProgress(null);
    setSelectedNode(null);
    try {
      const nextReport = await inspectModel(file, confirmLargeModel, setImportProgress);
      setReport(nextReport);
      setModelIndex(0);
      setGraphPath(nextReport.models[0]?.graphs[0]?.path ?? "");
      setLocation({ kind: "overview" });
      if (nextReport.models.length === 0) {
        setError(nextReport.errors[0]?.message ?? "未发现可读取的 ONNX 模型");
      }
    } catch (reason) {
      setReport(null);
      if (reason instanceof MemoryConfirmationError) {
        setMemoryWarning(reason.warning);
      } else {
        setError(reason instanceof Error ? reason.message : "模型导入失败");
      }
    } finally {
      setLoading(false);
      setImportProgress(null);
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
    const nextGraph = report?.models[index]?.graphs[0];
    setModelIndex(index);
    setGraphPath(nextGraph?.path ?? "");
    if (nextGraph) {
      setLocation({ kind: "scope", model_index: index, graph_path: nextGraph.path, scope: [] });
    }
    setSelectedNode(null);
  };

  const navigate = (nextLocation: HierarchyLocation) => {
    setLocation(nextLocation);
    setSelectedNode(null);
    if (nextLocation.kind !== "overview") {
      setModelIndex(nextLocation.model_index);
      setGraphPath(nextLocation.graph_path);
    }
  };

  const selectOperator = (operator: OperatorReport) => {
    setGraphPath(operator.graph_path);
    const operatorGraph = model?.graphs.find((item) => item.path === operator.graph_path);
    if (operatorGraph) {
      setLocation({
        kind: "scope",
        model_index: modelIndex,
        graph_path: operator.graph_path,
        scope: scopeForOperator(operator),
      });
    }
    setSelectedNode({
      kind: "operator",
      label: operator.op_type,
      subtitle: operator.name || `${displayDomain(operator.domain)} / ${operator.index}`,
      inputs: operator.inputs,
      outputs: operator.outputs,
      input_values: operatorGraph ? tensorValues(operatorGraph, operator.inputs) : [],
      output_values: operatorGraph ? tensorValues(operatorGraph, operator.outputs) : [],
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
                {selectedFile
                  ? `${formatBytes(selectedFile.size)} / ${report?.source_type ?? "..."}`
                  : "ONNX / ARCHIVE"}
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
            accept={[
              ".onnx,.onnx.gz,.onnx.bz2,.onnx.xz",
              ".zip,.tar,.tar.gz,.tgz,.tar.bz2,.tbz2,.tar.xz,.txz,.7z,.rar",
              "application/zip,application/gzip,application/x-7z-compressed",
              "application/vnd.rar,application/x-rar-compressed,application/x-tar",
              "application/octet-stream",
            ].join(",")}
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
                className={index === modelIndex && location.kind !== "overview" ? "is-active" : ""}
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
            <b>{String(operatorGroups.length).padStart(3, "0")}</b>
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
            {filteredOperatorGroups.map((group) => (
              <button
                type="button"
                key={group.key}
                className={selectedNode?.operator
                  && operatorGroupKey(selectedNode.operator) === group.key ? "is-active" : ""}
                onClick={() => selectOperator(group.representative)}
              >
                <Box size={14} strokeWidth={1.7} />
                <span>
                  <strong>{group.representative.op_type}</strong>
                  <small>{displayDomain(group.representative.domain)}</small>
                </span>
                <b className="operator-count" aria-label={`${group.instances.length} instances`}>
                  {group.instances.length}
                </b>
                <ChevronRight size={13} />
              </button>
            ))}
            {model && filteredOperatorGroups.length === 0 && (
              <span className="list-empty">NO MATCH</span>
            )}
          </div>
        </section>
      </aside>

      <section className="workspace">
        <div className="workspace-bar">
          <div className="graph-breadcrumbs" aria-label="算图层级">
            {hierarchyView.breadcrumbs.map((item, index) => (
              <span key={`${item.label}-${index}`}>
                {index > 0 && <ChevronRight size={12} aria-hidden="true" />}
                <button type="button" onClick={() => navigate(item.location)}>
                  {item.label}
                </button>
              </span>
            ))}
            {hierarchyView.breadcrumbs.length === 0 && <strong>UNLOADED</strong>}
          </div>
          {model && location.kind !== "overview" && (
            <label className="graph-select">
              <span>GRAPH</span>
              <select
                value={graph?.path ?? ""}
                onChange={(event) => {
                  setGraphPath(event.target.value);
                  setLocation({
                    kind: "scope",
                    model_index: modelIndex,
                    graph_path: event.target.value,
                    scope: [],
                  });
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
          <GraphCanvas
            content={hierarchyView.content}
            onSelectNode={setSelectedNode}
            onOpenNode={(node) => {
              if (node.navigation) navigate(node.navigation);
            }}
          />
          {loading && (
            <div className="loading-screen" role="status">
              <LoaderCircle size={22} className="spin" />
              <strong>{progressLabel(importProgress)}</strong>
              <span title={importProgress?.model_path}>{importProgress?.model_path ?? selectedFile?.name}</span>
              <div className="import-progress" aria-label="导入进度">
                <i style={{ width: `${importProgress?.percent ?? 1}%` }} />
              </div>
              <small>
                {importProgress?.total
                  ? `${importProgress.current || importProgress.completed} / ${importProgress.total}`
                    + ` · ${importProgress.percent.toFixed(0)}%`
                  : "正在读取来源"}
              </small>
            </div>
          )}
        </div>
      </section>

      <aside className="inspector">
        <div className="section-heading">
          <span>INSPECTOR</span>
          <b>03</b>
        </div>
        <NodeInspector
          node={selectedNode}
          model={hierarchyView.model}
          graph={hierarchyView.graph}
          report={report}
        />
      </aside>

      <footer className="statusbar">
        <span className={report ? "status-ready" : ""}>{report ? "READY" : "IDLE"}</span>
        <span>MODELS {report?.models.length ?? 0}</span>
        <span>OPS {report?.operator_count ?? 0}</span>
        <span>ERRORS {report?.errors.length ?? 0}</span>
        <b>VULKAN TARGET / OFFLINE</b>
      </footer>
      {memoryWarning && selectedFile && (
        <div className="dialog-backdrop" role="presentation">
          <section
            className="memory-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="memory-dialog-title"
          >
            <div className="memory-dialog__signal" aria-hidden="true">
              <AlertTriangle size={24} />
            </div>
            <div className="memory-dialog__content">
              <span>MEMORY LIMIT WARNING</span>
              <h2 id="memory-dialog-title">模型过大，还要加载吗？</h2>
              <p>
                <strong>{memoryWarning.model_path}</strong> 预计需要 {formatBytes(memoryWarning.estimated_bytes)}，
                已超过当前可用内存 {formatBytes(memoryWarning.available_bytes)} 的 60% 警戒线。
              </p>
            </div>
            <div className="memory-dialog__actions">
              <button type="button" onClick={() => setMemoryWarning(null)}>
                取消加载
              </button>
              <button
                type="button"
                className="is-danger"
                onClick={() => void loadFile(selectedFile, true)}
              >
                我知道我在做什么
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

function progressLabel(progress: ImportProgress | null): string {
  if (!progress) return "正在上传模型";
  if (progress.phase === "scanning") return "正在扫描归档";
  if (progress.phase === "discovered") return `发现 ${progress.total} 个 ONNX 模型`;
  if (progress.phase === "reading") return "正在读取 ONNX 模型";
  if (progress.phase === "extracting") return "正在解压 ONNX 模型";
  if (progress.phase === "parsing") return "正在解析算子图";
  return "模型解析完成";
}

function NodeInspector({
  node,
  model,
  graph,
  report,
}: {
  node: CanvasNodeData | null;
  model: ModelReport | null;
  graph: GraphReport | null;
  report: InspectionReport | null;
}) {
  if (!node) {
    if (!model && report) {
      return (
        <div className="inspector-empty">
          <Cpu size={24} strokeWidth={1.4} />
          <strong>MODEL PIPELINE</strong>
          <dl>
            <div><dt>MODELS</dt><dd>{report.models.length}</dd></div>
            <div><dt>OPERATORS</dt><dd>{report.operator_count}</dd></div>
            <div><dt>ERRORS</dt><dd>{report.errors.length}</dd></div>
          </dl>
        </div>
      );
    }
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
      {!node.operator && node.node_count && (
        <dl className="property-list">
          <div><dt>LEVEL</dt><dd>{node.kind.toUpperCase()}</dd></div>
          <div><dt>OPERATORS</dt><dd>{node.node_count}</dd></div>
          {node.loop_state_count && (
            <div><dt>LOOP STATE</dt><dd>{node.loop_state_count} TENSORS</dd></div>
          )}
          {node.loop_condition && (
            <div><dt>EXIT</dt><dd title={node.loop_condition}>{node.loop_condition}</dd></div>
          )}
          <div><dt>ACTION</dt><dd className="pending">DOUBLE CLICK</dd></div>
        </dl>
      )}
      <TensorList title="INPUT TENSORS" tensors={node.input_values} />
      <TensorList title="OUTPUT TENSORS" tensors={node.output_values} />
    </div>
  );
}

function formatTensorShape(tensor: TensorValueReport): string {
  if (tensor.data_type === "UNKNOWN" && tensor.shape.length === 0) return "UNKNOWN";
  return tensor.shape.length === 0 ? "[]" : `[${tensor.shape.join(" × ")}]`;
}

function TensorList({ title, tensors }: { title: string; tensors: TensorValueReport[] }) {
  return (
    <section className="tensor-list">
      <h3>{title}<b>{tensors.length}</b></h3>
      {tensors.map((tensor, index) => (
        <div key={`${tensor.name}-${index}`}>
          <span>{String(index).padStart(2, "0")}</span>
          <span className="tensor-list__value">
            <code title={tensor.name}>{tensor.name || "OPTIONAL"}</code>
            <small>
              <b>{tensor.data_type}</b>
              <i>{formatTensorShape(tensor)}</i>
            </small>
          </span>
        </div>
      ))}
      {tensors.length === 0 && <span className="list-empty">NONE</span>}
    </section>
  );
}
