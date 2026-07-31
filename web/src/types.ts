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

export interface TensorValueReport {
  name: string;
  data_type: string;
  shape: string[];
}

export interface GraphReport {
  path: string;
  name: string;
  inputs: string[];
  outputs: string[];
  values: TensorValueReport[];
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
  source_type: "onnx" | "zip" | "tar" | "7z" | "rar" | "gzip" | "bzip2" | "xz";
  models: ModelReport[];
  errors: ModelError[];
  operator_count: number;
  operator_summary: OperatorCount[];
}

export interface MemoryWarning {
  code: "memory_confirmation_required";
  model_path: string;
  estimated_bytes: number;
  available_bytes: number;
  threshold_bytes: number;
  warning_ratio: number;
}

export interface ImportProgress {
  phase: "scanning" | "discovered" | "reading" | "extracting" | "parsing" | "completed";
  model_path: string;
  completed: number;
  current: number;
  total: number;
  percent: number;
}

export type CanvasNodeKind =
  | "input"
  | "module"
  | "group"
  | "operator"
  | "output"
  | "loop-zone";

export type HierarchyLocation =
  | { kind: "overview" }
  | {
      kind: "scope";
      model_index: number;
      graph_path: string;
      scope: string[];
    }
  | {
      kind: "operators";
      model_index: number;
      graph_path: string;
      operator_indices: number[];
      label: string;
      parent_scope: string[];
    };

export type CanvasNodeData = {
  kind: CanvasNodeKind;
  label: string;
  subtitle: string;
  inputs: string[];
  outputs: string[];
  input_values: TensorValueReport[];
  output_values: TensorValueReport[];
  operator?: OperatorReport;
  navigation?: HierarchyLocation;
  node_count?: number;
  inference?: "exact" | "inferred";
  loop_state_count?: number;
  loop_condition?: string;
  is_cyclic?: boolean;
  cycle_size?: number;
} & Record<string, unknown>;
