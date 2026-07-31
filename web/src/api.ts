import type { ImportProgress, InspectionReport, MemoryWarning } from "./types";

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
