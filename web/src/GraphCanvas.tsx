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
  type EdgeTypes,
  type NodeTypes,
} from "@xyflow/react";

import { FeedbackEdge } from "./FeedbackEdge";
import type { BuiltGraph, CanvasNode } from "./graph";
import { GraphNode } from "./GraphNode";
import { LoopZone } from "./LoopZone";
import type { CanvasNodeData } from "./types";

const nodeTypes: NodeTypes = { inspector: GraphNode, loopZone: LoopZone };
const edgeTypes: EdgeTypes = { feedback: FeedbackEdge };

interface GraphCanvasProps {
  content: BuiltGraph | null;
  onSelectNode: (node: CanvasNodeData | null) => void;
  onOpenNode: (node: CanvasNodeData) => void;
}

function GraphCanvasInner({ content, onSelectNode, onOpenNode }: GraphCanvasProps) {
  const initial = useMemo(
    () => content ?? { nodes: [], edges: [] },
    [content],
  );
  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasNode>(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const fitGraph = useCallback(
    (duration = 250) =>
      fitView({
        padding: 0.14,
        duration,
        minZoom: window.innerWidth <= 700 ? 0.08 : 0.12,
        maxZoom: 1,
      }),
    [fitView],
  );

  useEffect(() => {
    setNodes(initial.nodes);
    setEdges(initial.edges);
    window.setTimeout(() => void fitGraph(), 0);
  }, [fitGraph, initial, setEdges, setNodes]);

  if (!content) {
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
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => onSelectNode(node.data)}
      onNodeDoubleClick={(_, node) => onOpenNode(node.data)}
      onPaneClick={() => onSelectNode(null)}
      nodesConnectable={false}
      minZoom={0.06}
      maxZoom={2}
      fitView
      fitViewOptions={{
        padding: 0.14,
        minZoom: window.innerWidth <= 700 ? 0.08 : 0.12,
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
          return kind === "output" ? "#ba8b43" : kind === "input" ? "#59d98e" : "#62d8d3";
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
