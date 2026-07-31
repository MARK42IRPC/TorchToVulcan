export interface OpsetReport {
  domain: string;
  version: number;
}

export interface OperatorReport {
  graph_path: string;
  index: number;
  name: string;
  op_type: string;
  domain: string;
  inputs: string[];
  outputs: string[];
}

export interface GraphReport {
  path: string;
  name: string;
  inputs: string[];
  outputs: string[];
  operators: OperatorReport[];
}

export interface ModelReport {
  path: string;
  graph_name: string;
  ir_version: number;
  producer_name: string;
  producer_version: string;
  opsets: OpsetReport[];
  graphs: GraphReport[];
  operator_count: number;
}

export interface ModelError {
  path: string;
  message: string;
}

export interface OperatorCount {
  domain: string;
  op_type: string;
  count: number;
}

export interface InspectionReport {
  source: string;
  source_type: "onnx" | "zip" | "tar" | "7z" | "gzip" | "bzip2" | "xz";
  models: ModelReport[];
  errors: ModelError[];
  operator_count: number;
  operator_summary: OperatorCount[];
}

export type CanvasNodeKind = "input" | "operator" | "output";

export type CanvasNodeData = {
  kind: CanvasNodeKind;
  label: string;
  subtitle: string;
  inputs: string[];
  outputs: string[];
  operator?: OperatorReport;
} & Record<string, unknown>;
