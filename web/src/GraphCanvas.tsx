import { useCallback, useEffect, useMemo } from "react";
import { Focus, Minus, Plus } from "lucide-react";
import {
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type NodeTypes,
} from "@xyflow/react";

import { buildGraph, type CanvasNode } from "./graph";
import { GraphNode } from "./GraphNode";
import type { CanvasNodeData, GraphReport } from "./types";

const nodeTypes: NodeTypes = { inspector: GraphNode };

interface GraphCanvasProps {
  graph: GraphReport | null;
  onSelectNode: (node: CanvasNodeData | null) => void;
}

function GraphCanvasInner({ graph, onSelectNode }: GraphCanvasProps) {
  const initial = useMemo(() => (graph ? buildGraph(graph) : { nodes: [], edges: [] }), [graph]);
  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasNode>(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const fitGraph = useCallback(
    (duration = 250) =>
      fitView({
        padding: 0.14,
        duration,
        minZoom: window.innerWidth <= 700 ? 0.32 : 0.52,
        maxZoom: 1,
      }),
    [fitView],
  );

  useEffect(() => {
    setNodes(initial.nodes);
    setEdges(initial.edges);
    onSelectNode(null);
    window.setTimeout(() => void fitGraph(), 0);
  }, [fitGraph, initial, onSelectNode, setEdges, setNodes]);

  if (!graph) {
    return (
      <div className="canvas-empty">
        <div className="canvas-empty__mark" aria-hidden="true" />
        <strong>NO GRAPH LOADED</strong>
        <span>ONNX / ZIP</span>
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => onSelectNode(node.data)}
      onPaneClick={() => onSelectNode(null)}
      nodesConnectable={false}
      minZoom={0.25}
      maxZoom={2}
      fitView
      fitViewOptions={{
        padding: 0.14,
        minZoom: window.innerWidth <= 700 ? 0.32 : 0.52,
        maxZoom: 1,
      }}
      colorMode="dark"
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#243034" gap={24} size={1} variant={BackgroundVariant.Dots} />
      <MiniMap
        pannable
        zoomable
        nodeColor={(node) => {
          const kind = (node.data as CanvasNodeData).kind;
          return kind === "output" ? "#ba8b43" : kind === "input" ? "#497c7b" : "#62d8d3";
        }}
        maskColor="rgba(5, 8, 9, 0.78)"
      />
      <Panel position="top-right" className="canvas-tools">
        <button type="button" title="放大" aria-label="放大" onClick={() => void zoomIn()}>
          <Plus size={18} />
        </button>
        <button type="button" title="缩小" aria-label="缩小" onClick={() => void zoomOut()}>
          <Minus size={18} />
        </button>
        <button
          type="button"
          title="适应画布"
          aria-label="适应画布"
          onClick={() => void fitGraph()}
        >
          <Focus size={18} />
        </button>
      </Panel>
    </ReactFlow>
  );
}

export function GraphCanvas(props: GraphCanvasProps) {
  return (
    <ReactFlowProvider>
      <GraphCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
