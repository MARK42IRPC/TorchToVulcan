import { MarkerType, type Edge, type Node } from "@xyflow/react";

import type { CanvasNodeData, GraphReport, TensorValueReport } from "./types";

export type CanvasNode = Node<CanvasNodeData, "inspector" | "loopZone">;
export type BuiltGraph = { nodes: CanvasNode[]; edges: Edge[] };

type Producer = {
  nodeId: string;
  outputIndex: number;
};

type EdgeSpec = {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
};

type TopologyLayout = {
  positions: Map<string, { x: number; y: number }>;
  componentByNode: Map<string, number>;
  cyclicComponents: Set<number>;
  componentSizes: Map<number, number>;
};

const COLUMN_WIDTH = 294;
const ROW_HEIGHT = 148;
const COMPONENT_GAP = 24;
const CANVAS_MARGIN_X = 72;
const CANVAS_MARGIN_Y = 72;

function inputNodeId(graphPath: string, tensor: string): string {
  return `input:${graphPath}:${tensor}`;
}

function operatorNodeId(graphPath: string, index: number): string {
  return `operator:${graphPath}:${index}`;
}

function outputNodeId(graphPath: string, tensor: string): string {
  return `output:${graphPath}:${tensor}`;
}

export function buildGraph(graph: GraphReport): BuiltGraph {
  const nodes: CanvasNode[] = [];
  const edgeSpecs: EdgeSpec[] = [];
  const tensorProducer = new Map<string, Producer>();
  const inputIds = new Set<string>();
  const outputIds = new Set<string>();
  const values = new Map((graph.values ?? []).map((value) => [value.name, value]));
  const tensorValue = (name: string): TensorValueReport => values.get(name) ?? {
    name,
    data_type: "UNKNOWN",
    shape: [],
  };

  graph.inputs.forEach((tensor) => {
    const id = inputNodeId(graph.path, tensor);
    inputIds.add(id);
    tensorProducer.set(tensor, { nodeId: id, outputIndex: 0 });
    nodes.push({
      id,
      type: "inspector",
      position: { x: 0, y: 0 },
      data: {
        kind: "input",
        label: tensor,
        subtitle: "GRAPH INPUT",
        inputs: [],
        outputs: [tensor],
        input_values: [],
        output_values: [tensorValue(tensor)],
      },
    });
  });

  graph.operators.forEach((operator) => {
    const id = operatorNodeId(graph.path, operator.index);
    nodes.push({
      id,
      type: "inspector",
      position: { x: 0, y: 0 },
      data: {
        kind: "operator",
        label: operator.op_type,
        subtitle: operator.name || `${operator.domain || "ai.onnx"} / ${operator.index}`,
        inputs: operator.inputs,
        outputs: operator.outputs,
        input_values: operator.inputs.map(tensorValue),
        output_values: operator.outputs.map(tensorValue),
        operator,
      },
    });
  });

  // Collect all producers before resolving inputs. ONNX node arrays are usually
  // topological, but importer and hand-authored graphs do not need to rely on it.
  graph.operators.forEach((operator) => {
    const id = operatorNodeId(graph.path, operator.index);
    operator.outputs.forEach((tensor, outputIndex) => {
      if (tensor) tensorProducer.set(tensor, { nodeId: id, outputIndex });
    });
  });

  graph.operators.forEach((operator) => {
    const target = operatorNodeId(graph.path, operator.index);
    operator.inputs.forEach((tensor, inputIndex) => {
      const source = tensorProducer.get(tensor);
      if (!source || !tensor) return;
      edgeSpecs.push({
        id: `edge:${source.nodeId}:${source.outputIndex}:${target}:${inputIndex}`,
        source: source.nodeId,
        target,
        sourceHandle: `out-${source.outputIndex}`,
        targetHandle: `in-${inputIndex}`,
      });
    });
  });

  graph.outputs.forEach((tensor) => {
    const source = tensorProducer.get(tensor);
    const id = outputNodeId(graph.path, tensor);
    outputIds.add(id);
    nodes.push({
      id,
      type: "inspector",
      position: { x: 0, y: 0 },
      data: {
        kind: "output",
        label: tensor,
        subtitle: "GRAPH OUTPUT",
        inputs: [tensor],
        outputs: [],
        input_values: [tensorValue(tensor)],
        output_values: [],
      },
    });
    if (source) {
      edgeSpecs.push({
        id: `edge:${source.nodeId}:${source.outputIndex}:${id}:0`,
        source: source.nodeId,
        target: id,
        sourceHandle: `out-${source.outputIndex}`,
        targetHandle: "in-0",
      });
    }
  });

  const layout = layoutTopology(nodes, edgeSpecs, inputIds, outputIds);
  const positionedNodes = nodes.map((node) => {
    const component = layout.componentByNode.get(node.id);
    const isStronglyConnected = component !== undefined && layout.cyclicComponents.has(component);
    const isControlFlowLoop = ["Loop", "Scan"].includes(node.data.operator?.op_type ?? "");
    const isCyclic = isStronglyConnected || isControlFlowLoop;
    return {
      ...node,
      position: layout.positions.get(node.id) ?? { x: CANVAS_MARGIN_X, y: CANVAS_MARGIN_Y },
      data: {
        ...node.data,
        is_cyclic: isCyclic,
        cycle_size: isStronglyConnected ? layout.componentSizes.get(component) : isCyclic ? 1 : undefined,
      },
    };
  });

  const edges = edgeSpecs.map<Edge>((edge) => {
    const sourceComponent = layout.componentByNode.get(edge.source);
    const targetComponent = layout.componentByNode.get(edge.target);
    const isCycleEdge =
      sourceComponent !== undefined &&
      sourceComponent === targetComponent &&
      layout.cyclicComponents.has(sourceComponent);
    const isOutputEdge = outputIds.has(edge.target);
    const isInputEdge = inputIds.has(edge.source);
    const color = isCycleEdge
      ? "#f0b35a"
      : isOutputEdge
        ? "#ba8b43"
        : isInputEdge
          ? "#59d98e"
          : "#4fb9b6";
    return {
      ...edge,
      type: "bezier",
      className: isCycleEdge ? "graph-edge graph-edge--cycle" : "graph-edge",
      markerEnd: { type: MarkerType.ArrowClosed, color },
      style: {
        stroke: color,
        strokeWidth: isCycleEdge ? 1.8 : 1.35,
        strokeDasharray: isCycleEdge ? "7 5" : undefined,
      },
    };
  });

  return { nodes: positionedNodes, edges };
}

function layoutTopology(
  nodes: CanvasNode[],
  edges: EdgeSpec[],
  inputIds: Set<string>,
  outputIds: Set<string>,
): TopologyLayout {
  const nodeOrder = new Map(nodes.map((node, index) => [node.id, index]));
  const outgoing = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  edges.forEach((edge) => outgoing.get(edge.source)?.add(edge.target));

  const components = stronglyConnectedComponents(nodes.map((node) => node.id), outgoing, nodeOrder);
  const componentByNode = new Map<string, number>();
  components.forEach((members, component) => {
    members.forEach((nodeId) => componentByNode.set(nodeId, component));
  });

  const predecessors = new Map<number, Set<number>>();
  const successors = new Map<number, Set<number>>();
  const componentOrder = new Map<number, number>();
  const cyclicComponents = new Set<number>();
  components.forEach((members, component) => {
    predecessors.set(component, new Set());
    successors.set(component, new Set());
    componentOrder.set(
      component,
      Math.min(...members.map((nodeId) => nodeOrder.get(nodeId) ?? Number.MAX_SAFE_INTEGER)),
    );
    if (members.length > 1) cyclicComponents.add(component);
  });

  edges.forEach((edge) => {
    const source = componentByNode.get(edge.source);
    const target = componentByNode.get(edge.target);
    if (source === undefined || target === undefined) return;
    if (source === target) {
      if (edge.source === edge.target) cyclicComponents.add(source);
      return;
    }
    successors.get(source)?.add(target);
    predecessors.get(target)?.add(source);
  });

  const topologicalComponents = topologicalSort(
    components.map((_, component) => component),
    predecessors,
    successors,
    componentOrder,
  );
  const depthByComponent = new Map<number, number>();
  topologicalComponents.forEach((component) => {
    const members = components[component];
    const hasInput = members.some((nodeId) => inputIds.has(nodeId));
    const parentDepths = [...(predecessors.get(component) ?? [])].map(
      (parent) => depthByComponent.get(parent) ?? 0,
    );
    const depth = hasInput
      ? 0
      : Math.max(1, parentDepths.length > 0 ? Math.max(...parentDepths) + 1 : 1);
    depthByComponent.set(component, depth);
  });

  const operatorDepths = components
    .map((members, component) => ({ members, depth: depthByComponent.get(component) ?? 1 }))
    .filter(({ members }) => !members.every((nodeId) => inputIds.has(nodeId) || outputIds.has(nodeId)))
    .map(({ depth }) => depth);
  const outputDepth = Math.max(1, ...(operatorDepths.map((depth) => depth + 1)));
  components.forEach((members, component) => {
    if (members.some((nodeId) => outputIds.has(nodeId))) {
      depthByComponent.set(component, outputDepth);
    }
  });

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  [...topologicalComponents].reverse().forEach((component) => {
    const members = components[component];
    const isRootConstant =
      (predecessors.get(component)?.size ?? 0) === 0 &&
      members.every((nodeId) => nodeById.get(nodeId)?.data.operator?.op_type === "Constant");
    const childDepths = [...(successors.get(component) ?? [])].map(
      (child) => depthByComponent.get(child) ?? outputDepth,
    );
    if (isRootConstant && childDepths.length > 0) {
      depthByComponent.set(component, Math.max(1, Math.min(...childDepths) - 1));
    }
  });

  const layers = new Map<number, number[]>();
  components.forEach((_, component) => {
    const depth = depthByComponent.get(component) ?? 1;
    const layer = layers.get(depth) ?? [];
    layer.push(component);
    layers.set(depth, layer);
  });
  layers.forEach((layer) => {
    layer.sort((left, right) => (componentOrder.get(left) ?? 0) - (componentOrder.get(right) ?? 0));
  });
  minimizeCrossings(layers, predecessors, successors, componentOrder);

  const layerDepths = [...layers.keys()].sort((left, right) => left - right);
  const layerHeights = new Map<number, number>();
  layers.forEach((layer, depth) => {
    const membersHeight = layer.reduce(
      (height, component) => height + components[component].length * ROW_HEIGHT,
      0,
    );
    layerHeights.set(depth, membersHeight + Math.max(0, layer.length - 1) * COMPONENT_GAP);
  });
  const maxLayerHeight = Math.max(ROW_HEIGHT, ...layerHeights.values());
  const positions = new Map<string, { x: number; y: number }>();
  layerDepths.forEach((depth) => {
    const layer = layers.get(depth) ?? [];
    let y = CANVAS_MARGIN_Y + (maxLayerHeight - (layerHeights.get(depth) ?? 0)) / 2;
    layer.forEach((component) => {
      const members = [...components[component]].sort(
        (left, right) => (nodeOrder.get(left) ?? 0) - (nodeOrder.get(right) ?? 0),
      );
      members.forEach((nodeId, memberIndex) => {
        positions.set(nodeId, {
          x: CANVAS_MARGIN_X + depth * COLUMN_WIDTH,
          y: y + memberIndex * ROW_HEIGHT,
        });
      });
      y += members.length * ROW_HEIGHT + COMPONENT_GAP;
    });
  });

  return {
    positions,
    componentByNode,
    cyclicComponents,
    componentSizes: new Map(components.map((members, component) => [component, members.length])),
  };
}

function stronglyConnectedComponents(
  nodeIds: string[],
  outgoing: Map<string, Set<string>>,
  nodeOrder: Map<string, number>,
): string[][] {
  let nextIndex = 0;
  const indexByNode = new Map<string, number>();
  const lowLink = new Map<string, number>();
  const stack: string[] = [];
  const onStack = new Set<string>();
  const components: string[][] = [];

  const visit = (nodeId: string) => {
    indexByNode.set(nodeId, nextIndex);
    lowLink.set(nodeId, nextIndex);
    nextIndex += 1;
    stack.push(nodeId);
    onStack.add(nodeId);

    const targets = [...(outgoing.get(nodeId) ?? [])].sort(
      (left, right) => (nodeOrder.get(left) ?? 0) - (nodeOrder.get(right) ?? 0),
    );
    targets.forEach((target) => {
      if (!indexByNode.has(target)) {
        visit(target);
        lowLink.set(nodeId, Math.min(lowLink.get(nodeId) ?? 0, lowLink.get(target) ?? 0));
      } else if (onStack.has(target)) {
        lowLink.set(nodeId, Math.min(lowLink.get(nodeId) ?? 0, indexByNode.get(target) ?? 0));
      }
    });

    if (lowLink.get(nodeId) !== indexByNode.get(nodeId)) return;
    const component: string[] = [];
    let member: string;
    do {
      member = stack.pop() as string;
      onStack.delete(member);
      component.push(member);
    } while (member !== nodeId);
    component.sort((left, right) => (nodeOrder.get(left) ?? 0) - (nodeOrder.get(right) ?? 0));
    components.push(component);
  };

  nodeIds.forEach((nodeId) => {
    if (!indexByNode.has(nodeId)) visit(nodeId);
  });
  components.sort((left, right) =>
    (nodeOrder.get(left[0]) ?? 0) - (nodeOrder.get(right[0]) ?? 0));
  return components;
}

function topologicalSort(
  components: number[],
  predecessors: Map<number, Set<number>>,
  successors: Map<number, Set<number>>,
  componentOrder: Map<number, number>,
): number[] {
  const indegree = new Map(
    components.map((component) => [component, predecessors.get(component)?.size ?? 0]),
  );
  const ready = components
    .filter((component) => indegree.get(component) === 0)
    .sort((left, right) => (componentOrder.get(left) ?? 0) - (componentOrder.get(right) ?? 0));
  const result: number[] = [];
  while (ready.length > 0) {
    const component = ready.shift() as number;
    result.push(component);
    successors.get(component)?.forEach((successor) => {
      const nextIndegree = (indegree.get(successor) ?? 0) - 1;
      indegree.set(successor, nextIndegree);
      if (nextIndegree === 0) {
        ready.push(successor);
        ready.sort(
          (left, right) => (componentOrder.get(left) ?? 0) - (componentOrder.get(right) ?? 0),
        );
      }
    });
  }
  return result;
}

function minimizeCrossings(
  layers: Map<number, number[]>,
  predecessors: Map<number, Set<number>>,
  successors: Map<number, Set<number>>,
  componentOrder: Map<number, number>,
) {
  const depths = [...layers.keys()].sort((left, right) => left - right);
  const rowByComponent = new Map<number, number>();
  const updateRows = () => {
    layers.forEach((layer) => layer.forEach((component, row) => rowByComponent.set(component, row)));
  };
  const sweep = (orderedDepths: number[], neighbors: Map<number, Set<number>>) => {
    orderedDepths.forEach((depth) => {
      const layer = layers.get(depth) ?? [];
      const currentOrder = new Map(layer.map((component, row) => [component, row]));
      layer.sort((left, right) => {
        const leftScore = barycenter(neighbors.get(left), rowByComponent);
        const rightScore = barycenter(neighbors.get(right), rowByComponent);
        if (leftScore !== null && rightScore !== null && leftScore !== rightScore) {
          return leftScore - rightScore;
        }
        if (leftScore !== null && rightScore === null) return -1;
        if (leftScore === null && rightScore !== null) return 1;
        const currentDifference = (currentOrder.get(left) ?? 0) - (currentOrder.get(right) ?? 0);
        if (currentDifference !== 0) return currentDifference;
        return (componentOrder.get(left) ?? 0) - (componentOrder.get(right) ?? 0);
      });
      layer.forEach((component, row) => rowByComponent.set(component, row));
    });
  };

  updateRows();
  for (let iteration = 0; iteration < 6; iteration += 1) {
    sweep(depths.slice(1), predecessors);
    sweep([...depths].reverse().slice(1), successors);
  }
}

function barycenter(neighbors: Set<number> | undefined, rows: Map<number, number>): number | null {
  if (!neighbors || neighbors.size === 0) return null;
  const values = [...neighbors].map((neighbor) => rows.get(neighbor) ?? 0);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
