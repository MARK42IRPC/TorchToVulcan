import { Repeat2 } from "lucide-react";
import type { NodeProps } from "@xyflow/react";

import type { CanvasNode } from "./graph";

export function LoopZone({ data }: NodeProps<CanvasNode>) {
  return (
    <div className="loop-zone">
      <div className="loop-zone__header">
        <Repeat2 size={15} strokeWidth={1.8} aria-hidden="true" />
        <strong>AUTOREGRESSIVE LOOP</strong>
        <span>{data.loop_state_count ?? 0} STATE TENSORS</span>
      </div>
      <div className="loop-zone__entry">INITIAL / STATE IN</div>
      <div className="loop-zone__exit">
        EXIT · {data.loop_condition || "EXTERNAL CONDITION"}
      </div>
    </div>
  );
}
