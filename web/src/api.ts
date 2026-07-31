import type { InspectionReport } from "./types";

export async function inspectModel(file: File): Promise<InspectionReport> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/inspect", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Import failed with HTTP ${response.status}`);
  }

  return (await response.json()) as InspectionReport;
}

