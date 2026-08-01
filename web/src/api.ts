import type {
  ImportProgress,
  InspectionReport,
  MemoryWarning,
  VerificationEvent,
  VerificationSummary,
  VerificationTarget,
  VerificationTensor,
} from "./types";

export class MemoryConfirmationError extends Error {
  constructor(public readonly warning: MemoryWarning) {
    super("模型大小超过可用内存警戒线");
    this.name = "MemoryConfirmationError";
  }
}

export async function inspectModel(
  file: File,
  confirmLargeModel = false,
  onProgress?: (progress: ImportProgress) => void,
): Promise<InspectionReport> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("confirm_large_model", String(confirmLargeModel));

  const response = await fetch("/api/inspect/stream", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string | MemoryWarning;
    } | null;
    if (
      response.status === 409 &&
      typeof body?.detail === "object" &&
      body.detail.code === "memory_confirmation_required"
    ) {
      throw new MemoryConfirmationError(body.detail);
    }
    const detail = typeof body?.detail === "string" ? body.detail : null;
    throw new Error(detail ?? `Import failed with HTTP ${response.status}`);
  }

  if (!response.body) throw new Error("浏览器不支持流式导入响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let report: InspectionReport | null = null;

  const consumeLine = (line: string) => {
    if (!line.trim()) return;
    const event = JSON.parse(line) as
      | { type: "progress"; progress: ImportProgress }
      | { type: "result"; report: InspectionReport }
      | { type: "memory_warning"; warning: MemoryWarning }
      | { type: "error"; message: string };
    if (event.type === "progress") onProgress?.(event.progress);
    if (event.type === "result") report = event.report;
    if (event.type === "memory_warning") throw new MemoryConfirmationError(event.warning);
    if (event.type === "error") throw new Error(event.message);
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    lines.forEach(consumeLine);
    if (done) break;
  }
  consumeLine(buffer);
  if (!report) throw new Error("导入服务未返回检查结果");
  return report;
}

export function buildVerificationTargets(report: InspectionReport): VerificationTarget[] {
  const targets = new Map<string, VerificationTarget>();
  report.models.forEach((model) => {
    model.graphs.forEach((graph) => {
      const values = new Map((graph.values ?? []).map((value) => [value.name, value]));
      const tensor = (name: string): VerificationTensor => {
        const value = values.get(name);
        return {
          name,
          data_type: value?.data_type ?? "UNKNOWN",
          shape: value?.shape ?? [],
          shape_known: value?.shape_known ?? false,
        };
      };
      graph.operators.forEach((operator) => {
        const inputs = operator.inputs.filter(Boolean).map(tensor);
        const outputs = operator.outputs.filter(Boolean).map(tensor);
        const attributes = Object.fromEntries(
          (operator.attributes ?? []).map((attribute) => [attribute.name, attribute.value]),
        );
        const signature = JSON.stringify({
          domain: operator.domain,
          op_type: operator.op_type,
          opset: operator.opset_version ?? 0,
          attributes,
          inputs: inputs.map((item) => [item.data_type, item.shape_known, item.shape.length]),
          outputs: outputs.map((item) => [item.data_type, item.shape_known, item.shape.length]),
        });
        if (!targets.has(signature)) {
          targets.set(signature, {
            target_id: `${model.path}:${operator.graph_path}:${operator.index}`,
            semantic_key: operator.semantics_key ?? "",
            domain: operator.domain,
            op_type: operator.op_type,
            opset_version: operator.opset_version ?? 0,
            attributes,
            inputs,
            outputs,
          });
        }
      });
    });
  });
  return [...targets.values()];
}

export async function verifyMappings(
  report: InspectionReport,
  onEvent: (event: VerificationEvent) => void,
): Promise<VerificationSummary> {
  const response = await fetch("/api/verify/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ targets: buildVerificationTargets(report) }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Verification failed with HTTP ${response.status}`);
  }
  if (!response.body) throw new Error("浏览器不支持流式验证响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let summary: VerificationSummary | null = null;
  const consumeLine = (line: string) => {
    if (!line.trim()) return;
    const event = JSON.parse(line) as VerificationEvent;
    onEvent(event);
    if (event.type === "result") summary = event.summary;
    if (event.type === "error") throw new Error(event.message);
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    lines.forEach(consumeLine);
    if (done) break;
  }
  consumeLine(buffer);
  if (!summary) throw new Error("验证服务未返回汇总结果");
  return summary;
}
