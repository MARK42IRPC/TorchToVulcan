import { MarkerType, type Edge, type Node } from "@xyflow/react";

import type { CanvasNodeData, GraphReport } from "./types";

export type CanvasNode = Node<CanvasNodeData, "inspector">;

const COLUMN_WIDTH = 244;
const ROW_HEIGHT = 126;

function inputNodeId(graphPath: string, tensor: string): string {
  return `input:${graphPath}:${tensor}`;
}

function operatorNodeId(graphPath: string, index: number): string {
  return `operator:${graphPath}:${index}`;
}

function outputNodeId(graphPath: string, tensor: string): string {
  return `output:${graphPath}:${tensor}`;
}

export function buildGraph(graph: GraphReport): { nodes: CanvasNode[]; edges: Edge[] } {
  const nodes: CanvasNode[] = [];
  const edges: Edge[] = [];
  const tensorProducer = new Map<string, string>();
  const nodeDepth = new Map<string, number>();

  graph.inputs.forEach((tensor) => {
    const id = inputNodeId(graph.path, tensor);
    tensorProducer.set(tensor, id);
    nodeDepth.set(id, 0);
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
      },
    });
  });

  graph.operators.forEach((operator) => {
    const id = operatorNodeId(graph.path, operator.index);
    const dependencyDepths = operator.inputs
      .map((tensor) => tensorProducer.get(tensor))
      .filter((producer): producer is string => producer !== undefined)
      .map((producer) => nodeDepth.get(producer) ?? 0);
    const depth = (dependencyDepths.length > 0 ? Math.max(...dependencyDepths) : 0) + 1;
    nodeDepth.set(id, depth);

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
        operator,
      },
    });

    operator.inputs.forEach((tensor, inputIndex) => {
      const source = tensorProducer.get(tensor);
      if (source) {
        edges.push({
          id: `edge:${source}:${id}:${inputIndex}`,
          source,
          target: id,
          type: "smoothstep",
          markerEnd: { type: MarkerType.ArrowClosed, color: "#65ddd8" },
          style: { stroke: "#4fb9b6", strokeWidth: 1.4 },
        });
      }
    });

    operator.outputs.forEach((tensor) => tensorProducer.set(tensor, id));
  });

  graph.outputs.forEach((tensor) => {
    const source = tensorProducer.get(tensor);
    const id = outputNodeId(graph.path, tensor);
    const depth = (source ? nodeDepth.get(source) ?? 0 : 0) + 1;
    nodeDepth.set(id, depth);
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
      },
    });
    if (source) {
      edges.push({
        id: `edge:${source}:${id}`,
        source,
        target: id,
        type: "smoothstep",
        markerEnd: { type: MarkerType.ArrowClosed, color: "#f0b35a" },
        style: { stroke: "#ba8b43", strokeWidth: 1.4 },
      });
    }
  });

  const rowsByDepth = new Map<number, number>();
  const positionedNodes = nodes.map((node) => {
    const depth = nodeDepth.get(node.id) ?? 0;
    const row = rowsByDepth.get(depth) ?? 0;
    rowsByDepth.set(depth, row + 1);
    return {
      ...node,
      position: {
        x: 72 + depth * COLUMN_WIDTH,
        y: 72 + row * ROW_HEIGHT,
      },
    };
  });

  return { nodes: positionedNodes, edges };
}

