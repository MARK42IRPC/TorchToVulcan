import { memo } from "react";
import {
  Box,
  Boxes,
  CircleArrowLeft,
  CircleArrowRight,
  Cpu,
  Repeat2,
} from "lucide-react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

import type { CanvasNode } from "./graph";

function typeSummary(types: string[]): string {
  const unique = [...new Set(types)];
  if (unique.length === 0) return "NONE";
  if (unique.length <= 2) return unique.join("+");
  return `${unique.slice(0, 2).join("+")}+${unique.length - 2}`;
}

function GraphNodeComponent({ data, selected }: NodeProps<CanvasNode>) {
  const Icon = data.kind === "input"
    ? CircleArrowRight
    : data.kind === "output"
      ? CircleArrowLeft
      : data.is_cyclic
        ? Repeat2
        : data.kind === "module"
          ? Cpu
          : data.kind === "group"
            ? Boxes
            : Box;
  const portTop = (index: number, count: number) => `${((index + 1) / (count + 1)) * 100}%`;
  const className = [
    "graph-node",
    `graph-node--${data.kind}`,
    data.is_cyclic ? "is-cyclic" : "",
    data.loop_state_count ? "is-loop-body" : "",
    data.navigation ? "is-openable" : "",
    selected ? "is-selected" : "",
  ].filter(Boolean).join(" ");
  const inputTypes = data.input_values.map((value) => value.data_type);
  const outputTypes = data.output_values.map((value) => value.data_type);

  return (
    <div className={className} title={data.navigation ? "双击进入下一层" : undefined}>
      {data.inputs.map((_, index) => (
        <Handle
          key={`in-${index}`}
          id={`in-${index}`}
          type="target"
          position={Position.Left}
          style={{ top: portTop(index, data.inputs.length) }}
        />
      ))}
      <div className="graph-node__rail" />
      <div className="graph-node__header">
        <Icon
          size={15}
          strokeWidth={1.8}
          aria-hidden="true"
        />
        <span>
          {data.loop_state_count ? "LOOP BODY" : data.kind.toUpperCase()}
          {data.navigation ? " / OPEN" : ""}
        </span>
      </div>
      <strong title={data.label}>{data.label}</strong>
      <span className="graph-node__subtitle" title={data.subtitle}>
        {data.subtitle}
      </span>
      <div
        className="graph-node__types"
        title={`IN ${inputTypes.join(", ") || "NONE"} / OUT ${outputTypes.join(", ") || "NONE"}`}
      >
        <span>IN {typeSummary(inputTypes)}</span>
        <span>OUT {typeSummary(outputTypes)}</span>
      </div>
      <div className="graph-node__ports">
        <span>IN {data.input_values.length}</span>
        <span>{data.node_count ? `${data.node_count} OPS` : `OUT ${data.output_values.length}`}</span>
      </div>
      {data.outputs.map((_, index) => (
        <Handle
          key={`out-${index}`}
          id={`out-${index}`}
          type="source"
          position={Position.Right}
          style={{ top: portTop(index, data.outputs.length) }}
        />
      ))}
    </div>
  );
}

export const GraphNode = memo(GraphNodeComponent);
