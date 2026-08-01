import type { Edge } from "@xyflow/react";

import { buildGraph, type BuiltGraph } from "./graph";
import type {
  CanvasNodeData,
  GraphReport,
  HierarchyLocation,
  InspectionReport,
  ModelReport,
  OperatorReport,
  TensorValueReport,
} from "./types";

const LEAF_OPERATOR_LIMIT = 80;
const OVERVIEW_PATH = "__model_pipeline__";
const valueMaps = new WeakMap<GraphReport, Map<string, TensorValueReport>>();

type Unit = {
  label: string;
  subtitle: string;
  kind: "group" | "operator";
  operatorIndices: number[];
  navigation?: HierarchyLocation;
  operator?: OperatorReport;
};

type Relation = {
  id: string;
  source: number | null;
  target: number | null;
  tensors: string[];
  confidence?: "exact" | "inferred";
  feedback?: boolean;
};

export type HierarchyBreadcrumb = {
  label: string;
  location: HierarchyLocation;
};

export type HierarchyView = {
  content: BuiltGraph | null;
  title: string;
  breadcrumbs: HierarchyBreadcrumb[];
  model: ModelReport | null;
  graph: GraphReport | null;
};

function nodeId(path: string, index: number): string {
  return `operator:${path}:${index}`;
}

function inputNodeId(path: string, tensor: string): string {
  return `input:${path}:${tensor}`;
}

function outputNodeId(path: string, tensor: string): string {
  return `output:${path}:${tensor}`;
}

function basename(path: string): string {
  return path.replaceAll("\\", "/").split("/").pop() ?? path;
}

function withoutOnnx(path: string): string {
  return basename(path).replace(/\.onnx$/i, "");
}

function valueMap(graph: GraphReport): Map<string, TensorValueReport> {
  let values = valueMaps.get(graph);
  if (!values) {
    values = new Map((graph.values ?? []).map((value) => [value.name, value]));
    valueMaps.set(graph, values);
  }
  return values;
}

function tensorValue(graph: GraphReport, name: string): TensorValueReport {
  return valueMap(graph).get(name) ?? {
    name,
    data_type: "UNKNOWN",
    shape: [],
    shape_known: false,
  };
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function summarizeBus(id: string, tensors: TensorValueReport[]): TensorValueReport {
  const types = unique(tensors.map((tensor) => tensor.data_type));
  const shapes = unique(tensors.map((tensor) => tensor.shape.join("\0")));
  return {
    name: id,
    data_type: types.length === 1 ? types[0] : "MULTI",
    shape: shapes.length === 1 ? tensors[0]?.shape ?? [] : [],
    shape_known: tensors.every((tensor) => tensor.shape_known !== false)
      && shapes.length === 1,
  };
}

function decorateNode(
  content: BuiltGraph,
  id: string,
  values: Partial<CanvasNodeData>,
): void {
  const node = content.nodes.find((candidate) => candidate.id === id);
  if (node) node.data = { ...node.data, ...values };
}

function connectionScore(output: TensorValueReport, input: TensorValueReport): number {
  if (
    output.data_type !== "UNKNOWN"
    && input.data_type !== "UNKNOWN"
    && output.data_type !== input.data_type
  ) return 0;

  const source = output.name.toLowerCase();
  const target = input.name.toLowerCase();
  if (source === target) return 100;
  if (
    source.startsWith("present_")
    && target.startsWith("past_")
    && source.slice(8) === target.slice(5)
  ) return 95;
  if (target === `i${source}`) return 90;
  if (source.length >= 4 && target.endsWith(`_${source}`)) return 80;
  return 0;
}

function feedbackScore(output: TensorValueReport, input: TensorValueReport): number {
  const score = connectionScore(output, input);
  if (score >= 90) return score;
  return 0;
}

export function buildModelOverview(report: InspectionReport): BuiltGraph {
  const rootGraphs = report.models.map((model) => model.graphs[0]);
  const inputValues = rootGraphs.map((graph) =>
    graph.inputs.map((name) => tensorValue(graph, name))
  );
  const outputValues = rootGraphs.map((graph) =>
    graph.outputs.map((name) => tensorValue(graph, name))
  );
  const matches = new Map<string, { source: number; output: TensorValueReport; score: number }>();

  inputValues.forEach((inputs, target) => {
    inputs.forEach((input) => {
      const candidates = outputValues.flatMap((outputs, source) =>
        source === target
          ? []
          : outputs.map((output) => ({
              source,
              output,
              score: connectionScore(output, input),
            })),
      ).filter((candidate) => candidate.score > 0)
        .sort((left, right) => right.score - left.score || left.source - right.source);
      if (candidates[0]) matches.set(`${target}\0${input.name}`, candidates[0]);
    });
  });

  const relationValues = new Map<string, TensorValueReport[]>();
  const relations = new Map<string, Relation>();
  const addRelation = (
    key: string,
    source: number | null,
    target: number | null,
    tensor: TensorValueReport,
    confidence?: "exact" | "inferred",
    feedback = false,
  ) => {
    let relation = relations.get(key);
    if (!relation) {
      relation = {
        id: `pipeline:${relations.size}`,
        source,
        target,
        tensors: [],
        confidence,
        feedback,
      };
      relations.set(key, relation);
      relationValues.set(relation.id, []);
    }
    relation.tensors.push(tensor.name);
    relationValues.get(relation.id)?.push(tensor);
    if (confidence === "inferred") relation.confidence = "inferred";
    if (feedback) relation.feedback = true;
  };

  const usedOutputs = new Set<string>();
  inputValues.forEach((inputs, target) => {
    inputs.forEach((input) => {
      const match = matches.get(`${target}\0${input.name}`);
      if (match) {
        const confidence = match.score === 100 ? "exact" : "inferred";
        addRelation(`${match.source}>${target}`, match.source, target, input, confidence);
        usedOutputs.add(`${match.source}\0${match.output.name}`);
      } else {
        addRelation(`external>${target}`, null, target, input);
      }
    });
  });
  inputValues.forEach((inputs, modelIndex) => {
    inputs.forEach((input) => {
      const feedback = outputValues[modelIndex]
        .map((output) => ({ output, score: feedbackScore(output, input) }))
        .filter((candidate) => candidate.score > 0)
        .sort((left, right) => right.score - left.score)[0];
      if (!feedback) return;
      addRelation(
        `${modelIndex}>${modelIndex}:feedback`,
        modelIndex,
        modelIndex,
        input,
        "inferred",
        true,
      );
      usedOutputs.add(`${modelIndex}\0${feedback.output.name}`);
    });
  });
  outputValues.forEach((outputs, source) => {
    outputs.forEach((output) => {
      if (!usedOutputs.has(`${source}\0${output.name}`)) {
        addRelation(`${source}>terminal`, source, null, output);
      }
    });
  });

  const relationList = [...relations.values()];
  const operators: OperatorReport[] = report.models.map((model, index) => ({
    graph_path: OVERVIEW_PATH,
    index,
    name: model.path,
    op_type: withoutOnnx(model.path),
    domain: "torch-to-vulcan.module",
    inputs: relationList.filter((relation) => relation.target === index).map((item) => item.id),
    outputs: relationList.filter((relation) => relation.source === index).map((item) => item.id),
  }));
  const graph: GraphReport = {
    path: OVERVIEW_PATH,
    name: "MODEL PIPELINE",
    inputs: relationList.filter((relation) => relation.source === null).map((item) => item.id),
    outputs: relationList.filter((relation) => relation.target === null).map((item) => item.id),
    values: relationList.map((relation) =>
      summarizeBus(relation.id, relationValues.get(relation.id) ?? [])
    ),
    operators,
  };
  const content = buildGraph(graph);

  report.models.forEach((model, index) => {
    const feedback = relationList.find((relation) =>
      relation.feedback && relation.source === index && relation.target === index
    );
    const condition = outputValues[index].find((value) =>
      /(?:stop|condition|finished|eos)/i.test(value.name)
    );
    decorateNode(content, nodeId(OVERVIEW_PATH, index), {
      kind: "module",
      label: withoutOnnx(model.path),
      subtitle: `${model.operator_count} OPS · ${model.graphs.length} GRAPH`,
      input_values: inputValues[index],
      output_values: outputValues[index],
      node_count: model.operator_count,
      loop_state_count: feedback?.tensors.length,
      loop_condition: condition?.name,
      navigation: {
        kind: "scope",
        model_index: index,
        graph_path: model.graphs[0].path,
        scope: [],
      },
    });
  });
  relationList.forEach((relation) => {
    const values = relationValues.get(relation.id) ?? [];
    const label = values.length === 1 ? values[0].name : `${values.length} TENSORS`;
    if (relation.source === null && relation.target !== null) {
      decorateNode(content, inputNodeId(OVERVIEW_PATH, relation.id), {
        label,
        subtitle: `EXTERNAL · ${withoutOnnx(report.models[relation.target].path)}`,
        output_values: values,
      });
    }
    if (relation.target === null && relation.source !== null) {
      decorateNode(content, outputNodeId(OVERVIEW_PATH, relation.id), {
        label,
        subtitle: `UNCONSUMED · ${withoutOnnx(report.models[relation.source].path)}`,
        input_values: values,
      });
    }
  });

  const inferredPairs = new Set(
    relationList
      .filter((relation) => relation.confidence === "inferred")
      .map((relation) => `${relation.source}>${relation.target}`),
  );
  content.edges = content.edges.map<Edge>((edge) => {
    const modulePrefix = `operator:${OVERVIEW_PATH}:`;
    if (!edge.source.startsWith(modulePrefix) || !edge.target.startsWith(modulePrefix)) {
      return edge;
    }
    const source = Number(edge.source.split(":").pop());
    const target = Number(edge.target.split(":").pop());
    if (!inferredPairs.has(`${source}>${target}`)) return edge;
    return {
      ...edge,
      className: "graph-edge graph-edge--inferred",
      markerEnd: typeof edge.markerEnd === "object"
        ? { ...edge.markerEnd, color: "#f0b35a" }
        : edge.markerEnd,
      style: { ...edge.style, stroke: "#f0b35a", strokeDasharray: "6 5" },
    };
  });
  const loopModels = relationList
    .filter((relation) => relation.feedback && relation.source !== null)
    .map((relation) => ({ modelIndex: relation.source as number, relation }));
  loopModels.forEach(({ modelIndex, relation }) => {
    const module = content.nodes.find((node) => node.id === nodeId(OVERVIEW_PATH, modelIndex));
    if (!module) return;
    content.nodes.forEach((node) => {
      if (node.position.y > module.position.y) node.position.y += 150;
    });
    const condition = outputValues[modelIndex].find((value) =>
      /(?:stop|condition|finished|eos)/i.test(value.name)
    );
    content.nodes.unshift({
      id: `loop-zone:${modelIndex}`,
      type: "loopZone",
      position: { x: module.position.x - 50, y: module.position.y - 66 },
      data: {
        kind: "loop-zone",
        label: "AUTOREGRESSIVE LOOP",
        subtitle: report.models[modelIndex].path,
        inputs: [],
        outputs: [],
        input_values: [],
        output_values: [],
        loop_state_count: relation.tensors.length,
        loop_condition: condition?.name,
      },
      style: { width: 312, height: 264 },
      draggable: false,
      selectable: false,
      connectable: false,
      focusable: false,
      zIndex: -10,
    });
    content.edges = content.edges.map((edge) => {
      if (edge.source !== module.id || edge.target !== module.id) return edge;
      return {
        ...edge,
        type: "feedback",
        className: "graph-edge graph-edge--feedback",
        data: { label: `ITERATION STATE × ${relation.tensors.length}` },
        style: { ...edge.style, stroke: "#f0b35a", strokeWidth: 2 },
        markerEnd: typeof edge.markerEnd === "object"
          ? { ...edge.markerEnd, color: "#f0b35a" }
          : edge.markerEnd,
      };
    });
  });
  return content;
}

export function scopeForOperator(operator: OperatorReport): string[] {
  const segments = operator.name.replaceAll("\\", "/").split("/").filter(Boolean);
  return segments.length > 1 ? segments.slice(0, -1) : [];
}

function startsWithScope(value: string[], scope: string[]): boolean {
  return scope.every((segment, index) => value[index] === segment);
}

function graphTopology(graph: GraphReport) {
  const producer = new Map<string, number>();
  const consumers = new Map<string, number[]>();
  const operatorByIndex = new Map<number, OperatorReport>();
  graph.operators.forEach((operator) => {
    operatorByIndex.set(operator.index, operator);
    operator.outputs.forEach((tensor) => {
      if (tensor) producer.set(tensor, operator.index);
    });
    operator.inputs.forEach((tensor) => {
      if (!tensor) return;
      const values = consumers.get(tensor) ?? [];
      values.push(operator.index);
      consumers.set(tensor, values);
    });
  });
  return { producer, consumers, operatorByIndex };
}

function buildOperatorSlice(graph: GraphReport, indices: number[], label: string): BuiltGraph {
  const selected = new Set(indices);
  const { producer, consumers } = graphTopology(graph);
  const operators = graph.operators.filter((operator) => selected.has(operator.index));
  const inputs = unique(operators.flatMap((operator) =>
    operator.inputs.filter((tensor) => tensor && !selected.has(producer.get(tensor) ?? -1))
  ));
  const outputs = unique(operators.flatMap((operator) =>
    operator.outputs.filter((tensor) => {
      const targets = consumers.get(tensor) ?? [];
      return graph.outputs.includes(tensor)
        || targets.length === 0
        || targets.some((target) => !selected.has(target));
    })
  ));
  return buildGraph({
    path: `${graph.path}::${label}:${indices[0] ?? 0}`,
    name: label,
    inputs,
    outputs,
    values: graph.values,
    operators,
  });
}

function buildAbstractScope(
  graph: GraphReport,
  units: Unit[],
  scope: string[],
): BuiltGraph {
  const path = `${graph.path}::scope:${scope.join("/") || "root"}`;
  const unitByOperator = new Map<number, number>();
  units.forEach((unit, unitIndex) => {
    unit.operatorIndices.forEach((operatorIndex) => unitByOperator.set(operatorIndex, unitIndex));
  });
  const universe = new Set(unitByOperator.keys());
  const { producer, consumers, operatorByIndex } = graphTopology(graph);
  const relations = new Map<string, Relation>();
  const addRelation = (source: number | null, target: number | null, tensor: string) => {
    const key = `${source ?? "external"}>${target ?? "terminal"}`;
    let relation = relations.get(key);
    if (!relation) {
      relation = { id: `scope:${relations.size}`, source, target, tensors: [] };
      relations.set(key, relation);
    }
    if (!relation.tensors.includes(tensor)) relation.tensors.push(tensor);
  };

  units.forEach((unit, targetUnit) => {
    unit.operatorIndices.forEach((operatorIndex) => {
      const operator = operatorByIndex.get(operatorIndex);
      operator?.inputs.forEach((tensor) => {
        if (!tensor) return;
        const sourceIndex = producer.get(tensor);
        const sourceUnit = sourceIndex === undefined ? undefined : unitByOperator.get(sourceIndex);
        if (sourceUnit !== targetUnit) addRelation(sourceUnit ?? null, targetUnit, tensor);
      });
    });
  });
  units.forEach((unit, sourceUnit) => {
    unit.operatorIndices.forEach((operatorIndex) => {
      const operator = operatorByIndex.get(operatorIndex);
      operator?.outputs.forEach((tensor) => {
        if (!tensor) return;
        const targets = consumers.get(tensor) ?? [];
        const outside = targets.some((target) => !universe.has(target));
        if (graph.outputs.includes(tensor) || outside || targets.length === 0) {
          addRelation(sourceUnit, null, tensor);
        }
      });
    });
  });

  const relationList = [...relations.values()];
  const synthetic: GraphReport = {
    path,
    name: scope.join("/") || graph.name,
    inputs: relationList.filter((relation) => relation.source === null).map((item) => item.id),
    outputs: relationList.filter((relation) => relation.target === null).map((item) => item.id),
    values: relationList.map((relation) => summarizeBus(
      relation.id,
      relation.tensors.map((tensor) => tensorValue(graph, tensor)),
    )),
    operators: units.map((unit, index) => ({
      graph_path: path,
      index,
      name: unit.label,
      op_type: unit.label,
      domain: "torch-to-vulcan.group",
      inputs: relationList.filter((relation) => relation.target === index).map((item) => item.id),
      outputs: relationList.filter((relation) => relation.source === index).map((item) => item.id),
    })),
  };
  const content = buildGraph(synthetic);
  units.forEach((unit, index) => {
    const incoming = relationList
      .filter((relation) => relation.target === index)
      .flatMap((relation) => relation.tensors.map((tensor) => tensorValue(graph, tensor)));
    const outgoing = relationList
      .filter((relation) => relation.source === index)
      .flatMap((relation) => relation.tensors.map((tensor) => tensorValue(graph, tensor)));
    decorateNode(content, nodeId(path, index), {
      kind: unit.kind,
      label: unit.label,
      subtitle: unit.subtitle,
      inputs: synthetic.operators[index].inputs,
      outputs: synthetic.operators[index].outputs,
      input_values: incoming,
      output_values: outgoing,
      operator: unit.operator,
      node_count: unit.operatorIndices.length > 1 ? unit.operatorIndices.length : undefined,
      navigation: unit.navigation,
    });
  });
  relationList.forEach((relation) => {
    const values = relation.tensors.map((tensor) => tensorValue(graph, tensor));
    const label = values.length === 1 ? values[0].name : `${values.length} TENSORS`;
    if (relation.source === null) {
      decorateNode(content, inputNodeId(path, relation.id), { label, output_values: values });
    }
    if (relation.target === null) {
      decorateNode(content, outputNodeId(path, relation.id), { label, input_values: values });
    }
  });
  return content;
}

function commonPrefix(values: string[][]): string[] {
  if (values.length === 0) return [];
  const result: string[] = [];
  for (let index = 0; ; index += 1) {
    const segment = values[0][index];
    if (segment === undefined || values.some((value) => value[index] !== segment)) break;
    result.push(segment);
  }
  return result;
}

function buildScopeContent(
  graph: GraphReport,
  modelIndex: number,
  scope: string[],
): BuiltGraph {
  const scoped = graph.operators.filter((operator) =>
    startsWithScope(scopeForOperator(operator), scope)
  );
  const direct = scoped.filter((operator) => scopeForOperator(operator).length === scope.length);
  const descendants = scoped.filter((operator) => scopeForOperator(operator).length > scope.length);
  const remainingScopes = descendants.map((operator) => scopeForOperator(operator).slice(scope.length));
  const shared = direct.length === 0 ? commonPrefix(remainingScopes) : [];
  const groups = new Map<string, OperatorReport[]>();
  descendants.forEach((operator) => {
    const remaining = scopeForOperator(operator).slice(scope.length);
    const key = shared.length > 0 ? shared.join("/") : remaining[0];
    const values = groups.get(key) ?? [];
    values.push(operator);
    groups.set(key, values);
  });

  if (groups.size === 0 && direct.length <= LEAF_OPERATOR_LIMIT) {
    return buildOperatorSlice(graph, direct.map((operator) => operator.index), scope.at(-1) ?? "root");
  }

  const units: Unit[] = [...groups.entries()].map(([label, operators]) => {
    const nextScope = [...scope, ...label.split("/")];
    return {
      label: label.replaceAll("/", " / "),
      subtitle: `${operators.length} OPS · SCOPE`,
      kind: "group",
      operatorIndices: operators.map((operator) => operator.index),
      navigation: {
        kind: "scope",
        model_index: modelIndex,
        graph_path: graph.path,
        scope: nextScope,
      },
    };
  });
  if (direct.length <= LEAF_OPERATOR_LIMIT) {
    direct.forEach((operator) => units.push({
      label: operator.op_type,
      subtitle: operator.name || `${operator.domain || "ai.onnx"} / ${operator.index}`,
      kind: "operator",
      operatorIndices: [operator.index],
      operator,
    }));
  } else {
    for (let offset = 0; offset < direct.length; offset += LEAF_OPERATOR_LIMIT) {
      const chunk = direct.slice(offset, offset + LEAF_OPERATOR_LIMIT);
      const label = `OPS ${String(offset).padStart(4, "0")}-${String(offset + chunk.length - 1).padStart(4, "0")}`;
      units.push({
        label,
        subtitle: `${chunk.length} OPS · TOPOLOGY RANGE`,
        kind: "group",
        operatorIndices: chunk.map((operator) => operator.index),
        navigation: {
          kind: "operators",
          model_index: modelIndex,
          graph_path: graph.path,
          operator_indices: chunk.map((operator) => operator.index),
          label,
          parent_scope: scope,
        },
      });
    }
  }
  return buildAbstractScope(graph, units, scope);
}

function breadcrumbs(
  report: InspectionReport,
  location: Exclude<HierarchyLocation, { kind: "overview" }>,
): HierarchyBreadcrumb[] {
  const model = report.models[location.model_index];
  const scope = location.kind === "scope" ? location.scope : location.parent_scope;
  const result: HierarchyBreadcrumb[] = [
    { label: "PIPELINE", location: { kind: "overview" } },
    {
      label: withoutOnnx(model.path),
      location: {
        kind: "scope",
        model_index: location.model_index,
        graph_path: location.graph_path,
        scope: [],
      },
    },
  ];
  scope.forEach((segment, index) => result.push({
    label: segment,
    location: {
      kind: "scope",
      model_index: location.model_index,
      graph_path: location.graph_path,
      scope: scope.slice(0, index + 1),
    },
  }));
  if (location.kind === "operators") result.push({ label: location.label, location });
  return result;
}

export function buildHierarchyView(
  report: InspectionReport | null,
  location: HierarchyLocation,
): HierarchyView {
  if (!report || report.models.length === 0) {
    return { content: null, title: "UNLOADED", breadcrumbs: [], model: null, graph: null };
  }
  if (location.kind === "overview") {
    return {
      content: buildModelOverview(report),
      title: "MODEL PIPELINE",
      breadcrumbs: [{ label: "PIPELINE", location }],
      model: null,
      graph: null,
    };
  }
  const model = report.models[location.model_index];
  const graph = model?.graphs.find((item) => item.path === location.graph_path)
    ?? model?.graphs[0]
    ?? null;
  if (!model || !graph) {
    return { content: null, title: "UNLOADED", breadcrumbs: [], model: null, graph: null };
  }
  const content = location.kind === "operators"
    ? buildOperatorSlice(graph, location.operator_indices, location.label)
    : buildScopeContent(graph, location.model_index, location.scope);
  const title = location.kind === "operators"
    ? location.label
    : location.scope.at(-1) ?? withoutOnnx(model.path);
  return { content, title, breadcrumbs: breadcrumbs(report, location), model, graph };
}
