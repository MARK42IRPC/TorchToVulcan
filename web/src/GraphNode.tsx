import { memo } from "react";
import { Box, CircleArrowLeft, CircleArrowRight } from "lucide-react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

import type { CanvasNode } from "./graph";

function GraphNodeComponent({ data, selected }: NodeProps<CanvasNode>) {
  const Icon =
    data.kind === "input" ? CircleArrowRight : data.kind === "output" ? CircleArrowLeft : Box;

  return (
    <div className={`graph-node graph-node--${data.kind}${selected ? " is-selected" : ""}`}>
      {data.kind !== "input" && <Handle type="target" position={Position.Left} />}
      <div className="graph-node__rail" />
      <div className="graph-node__header">
        <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
        <span>{data.kind.toUpperCase()}</span>
      </div>
      <strong title={data.label}>{data.label}</strong>
      <span className="graph-node__subtitle" title={data.subtitle}>
        {data.subtitle}
      </span>
      <div className="graph-node__ports">
        <span>IN {data.inputs.length}</span>
        <span>OUT {data.outputs.length}</span>
      </div>
      {data.kind !== "output" && <Handle type="source" position={Position.Right} />}
    </div>
  );
}

export const GraphNode = memo(GraphNodeComponent);

