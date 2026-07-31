import { BaseEdge, EdgeLabelRenderer, type EdgeProps } from "@xyflow/react";

export function FeedbackEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
  style,
  data,
}: EdgeProps) {
  const bottom = Math.max(sourceY, targetY) + 112;
  const path = [
    `M ${sourceX} ${sourceY}`,
    `C ${sourceX + 76} ${sourceY}, ${sourceX + 76} ${bottom}, ${sourceX} ${bottom}`,
    `L ${targetX} ${bottom}`,
    `C ${targetX - 76} ${bottom}, ${targetX - 76} ${targetY}, ${targetX} ${targetY}`,
  ].join(" ");

  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
      <EdgeLabelRenderer>
        <span
          className="feedback-edge__label"
          style={{ transform: `translate(-50%, -50%) translate(${(sourceX + targetX) / 2}px,${bottom}px)` }}
        >
          {String(data?.label ?? "ITERATION STATE")}
        </span>
      </EdgeLabelRenderer>
    </>
  );
}
