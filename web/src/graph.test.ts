import { describe, expect, test } from "vitest";

import { buildGraph } from "./graph";
import type { GraphReport, OperatorReport, TensorValueReport } from "./types";

function operator(
  index: number,
  name: string,
  inputs: string[],
  outputs: string[],
): OperatorReport {
  return {
    graph_path: "main",
    index,
    name,
    op_type: name,
    domain: "",
    inputs,
    outputs,
  };
}

function graph(
  inputs: string[],
  outputs: string[],
  operators: OperatorReport[],
  values: TensorValueReport[] = [],
): GraphReport {
  return { path: "main", name: "main", inputs, outputs, values, operators };
}

function nodeByLabel(result: ReturnType<typeof buildGraph>, label: string) {
  const node = result.nodes.find((candidate) => candidate.data.label === label);
  if (!node) throw new Error(`node not found: ${label}`);
  return node;
}

describe("topology layout", () => {
  test("lays out an out-of-order sequential graph from input to output", () => {
    const result = buildGraph(
      graph(
        ["input"],
        ["output"],
        [
          operator(1, "Relu", ["hidden"], ["output"]),
          operator(0, "MatMul", ["input"], ["hidden"]),
        ],
      ),
    );

    const input = nodeByLabel(result, "input");
    const matmul = nodeByLabel(result, "MatMul");
    const relu = nodeByLabel(result, "Relu");
    const output = nodeByLabel(result, "output");
    expect(input.position.x).toBeLessThan(matmul.position.x);
    expect(matmul.position.x).toBeLessThan(relu.position.x);
    expect(relu.position.x).toBeLessThan(output.position.x);
    expect(new Set(result.nodes.map((node) => node.position.y)).size).toBe(1);
    expect(result.edges).toHaveLength(3);
    expect(result.edges.every((edge) => edge.type === "bezier")).toBe(true);
    const inputEdge = result.edges.find((edge) => edge.source.startsWith("input:"));
    expect(inputEdge?.style?.stroke).toBe("#59d98e");
  });

  test("uses barycentric ordering to untangle a crossing branch", () => {
    const result = buildGraph(
      graph(
        ["left", "right"],
        ["merged"],
        [
          operator(0, "FromRight", ["right"], ["right_value"]),
          operator(1, "FromLeft", ["left"], ["left_value"]),
          operator(2, "Merge", ["left_value", "right_value"], ["merged"]),
        ],
      ),
    );

    expect(nodeByLabel(result, "FromLeft").position.y)
      .toBeLessThan(nodeByLabel(result, "FromRight").position.y);
    const positions = new Map(result.nodes.map((node) => [node.id, node.position]));
    result.edges.forEach((edge) => {
      expect(positions.get(edge.source)?.x).toBeLessThan(positions.get(edge.target)?.x ?? 0);
    });
  });

  test("collapses a strongly connected component and marks its loop edges", () => {
    const result = buildGraph(
      graph(
        ["seed"],
        ["a"],
        [
          operator(0, "LoopA", ["seed", "b"], ["a"]),
          operator(1, "LoopB", ["a"], ["b"]),
        ],
      ),
    );

    const loopA = nodeByLabel(result, "LoopA");
    const loopB = nodeByLabel(result, "LoopB");
    expect(loopA.data.is_cyclic).toBe(true);
    expect(loopB.data.is_cyclic).toBe(true);
    expect(loopA.data.cycle_size).toBe(2);
    expect(loopA.position.x).toBe(loopB.position.x);
    expect(loopA.position.y).not.toBe(loopB.position.y);
    expect(result.edges.filter((edge) => edge.className?.includes("cycle"))).toHaveLength(2);
    expect(nodeByLabel(result, "seed").position.x).toBeLessThan(loopA.position.x);
    expect(loopA.position.x).toBeLessThan(nodeByLabel(result, "a").position.x);
  });

  test("recognizes a self-loop without reversing forward edges", () => {
    const result = buildGraph(
      graph([], ["state"], [operator(0, "StateUpdate", ["state"], ["state"])]),
    );
    const update = nodeByLabel(result, "StateUpdate");
    const cycleEdge = result.edges.find((edge) => edge.source === edge.target);

    expect(update.data.is_cyclic).toBe(true);
    expect(update.data.cycle_size).toBe(1);
    expect(cycleEdge?.className).toContain("cycle");
    expect(update.position.x).toBeLessThan(nodeByLabel(result, "state").position.x);
  });

  test("places root constants next to their earliest consumer", () => {
    const result = buildGraph(
      graph(
        ["input"],
        ["output"],
        [
          operator(0, "Constant", [], ["weight"]),
          operator(1, "First", ["input"], ["first"]),
          operator(2, "Second", ["first"], ["second"]),
          operator(3, "Consumer", ["second", "weight"], ["output"]),
        ],
      ),
    );

    const constant = nodeByLabel(result, "Constant");
    const first = nodeByLabel(result, "First");
    const second = nodeByLabel(result, "Second");
    const consumer = nodeByLabel(result, "Consumer");
    expect(constant.position.x).toBe(second.position.x);
    expect(constant.position.x).toBeGreaterThan(first.position.x);
    expect(constant.position.x).toBeLessThan(consumer.position.x);
  });

  test("marks ONNX Loop and Scan control-flow operators as cyclic", () => {
    const result = buildGraph(
      graph(
        ["trip_count", "condition", "state"],
        ["output"],
        [operator(0, "Loop", ["trip_count", "condition", "state"], ["output"])],
      ),
    );

    const loop = nodeByLabel(result, "Loop");
    expect(loop.data.is_cyclic).toBe(true);
    expect(loop.data.cycle_size).toBe(1);
    expect(nodeByLabel(result, "trip_count").position.x).toBeLessThan(loop.position.x);
    expect(loop.position.x).toBeLessThan(nodeByLabel(result, "output").position.x);
  });

  test("attaches known and unknown tensor metadata to nodes", () => {
    const result = buildGraph(
      graph(
        ["input"],
        ["output"],
        [operator(0, "Relu", ["input"], ["output"])],
        [
          { name: "input", data_type: "FLOAT", shape: ["batch", "4"] },
          { name: "output", data_type: "FLOAT16", shape: ["batch", "4"] },
        ],
      ),
    );

    const relu = nodeByLabel(result, "Relu");
    expect(relu.data.input_values[0]).toEqual({
      name: "input",
      data_type: "FLOAT",
      shape: ["batch", "4"],
    });
    expect(relu.data.output_values[0].data_type).toBe("FLOAT16");
  });
});
