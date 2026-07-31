import { describe, expect, test } from "vitest";

import { buildHierarchyView, buildModelOverview } from "./hierarchy";
import type { GraphReport, InspectionReport, ModelReport, OperatorReport } from "./types";

function operator(
  index: number,
  name: string,
  opType: string,
  inputs: string[],
  outputs: string[],
): OperatorReport {
  return { graph_path: "main", index, name, op_type: opType, domain: "", inputs, outputs };
}

function graph(operators: OperatorReport[]): GraphReport {
  return {
    path: "main",
    name: "main",
    inputs: ["x"],
    outputs: ["result"],
    values: [
      { name: "x", data_type: "FLOAT", shape: ["1", "4"] },
      { name: "hidden", data_type: "FLOAT", shape: ["1", "4"] },
      { name: "attn", data_type: "FLOAT", shape: ["1", "4"] },
      { name: "result", data_type: "FLOAT", shape: ["1", "4"] },
    ],
    operators,
  };
}

function model(path: string, value: GraphReport): ModelReport {
  return {
    path,
    graph_name: value.name,
    ir_version: 10,
    producer_name: "torch",
    producer_version: "2.7",
    opsets: [{ domain: "", version: 18 }],
    graphs: [value],
    operator_count: value.operators.length,
  };
}

function report(models: ModelReport[]): InspectionReport {
  return {
    source: "models.zip",
    source_type: "zip",
    models,
    errors: [],
    operator_count: models.reduce((total, item) => total + item.operator_count, 0),
    operator_summary: [],
  };
}

describe("hierarchical graph", () => {
  test("connects model modules and marks naming-convention matches as inferred", () => {
    const encoderGraph: GraphReport = {
      path: "encoder",
      name: "encoder",
      inputs: ["tokens"],
      outputs: ["x", "present_k_layer_0"],
      values: [
        { name: "tokens", data_type: "INT64", shape: ["1", "n"] },
        { name: "x", data_type: "FLOAT", shape: ["1", "n", "4"] },
        { name: "present_k_layer_0", data_type: "FLOAT", shape: ["n", "1", "4"] },
      ],
      operators: [operator(0, "/encoder/MatMul", "MatMul", ["tokens"], ["x"])],
    };
    const decoderGraph: GraphReport = {
      path: "decoder",
      name: "decoder",
      inputs: ["x", "past_k_layer_0"],
      outputs: ["audio"],
      values: [
        { name: "x", data_type: "FLOAT", shape: ["1", "n", "4"] },
        { name: "past_k_layer_0", data_type: "FLOAT", shape: ["n", "1", "4"] },
        { name: "audio", data_type: "FLOAT", shape: ["samples"] },
      ],
      operators: [operator(0, "/decoder/Gemm", "Gemm", ["x"], ["audio"])],
    };

    const content = buildModelOverview(report([
      model("encoder.onnx", encoderGraph),
      model("decoder.onnx", decoderGraph),
    ]));
    const modules = content.nodes.filter((node) => node.data.kind === "module");
    const moduleEdge = content.edges.find((edge) =>
      edge.source.includes("operator:__model_pipeline__:0")
      && edge.target.includes("operator:__model_pipeline__:1")
    );

    expect(modules).toHaveLength(2);
    expect(modules[0].data.navigation?.kind).toBe("scope");
    expect(moduleEdge?.type).toBe("bezier");
    expect(moduleEdge?.className).toContain("inferred");
  });

  test("drills from shared scopes into exact leaf operators", () => {
    const value = graph([
      operator(0, "/encoder/layers.0/attn/MatMul", "MatMul", ["x"], ["hidden"]),
      operator(1, "/encoder/layers.0/attn/Softmax", "Softmax", ["hidden"], ["attn"]),
      operator(2, "/encoder/layers.0/mlp/Gemm", "Gemm", ["attn"], ["result"]),
    ]);
    const inspection = report([model("network.onnx", value)]);
    const root = buildHierarchyView(inspection, {
      kind: "scope",
      model_index: 0,
      graph_path: "main",
      scope: [],
    });
    const rootGroup = root.content?.nodes.find((node) => node.data.kind === "group");
    expect(rootGroup?.data.label).toBe("encoder / layers.0");

    const modules = buildHierarchyView(inspection, rootGroup?.data.navigation ?? { kind: "overview" });
    expect(
      modules.content?.nodes.filter((node) => node.data.kind === "group")
        .map((node) => node.data.label),
    ).toEqual(["attn", "mlp"]);

    const attention = modules.content?.nodes.find((node) => node.data.label === "attn");
    const leaf = buildHierarchyView(inspection, attention?.data.navigation ?? { kind: "overview" });
    expect(
      leaf.content?.nodes.filter((node) => node.data.kind === "operator")
        .map((node) => node.data.label),
    ).toEqual(["MatMul", "Softmax"]);
    expect(leaf.content?.edges.every((edge) => edge.type === "bezier")).toBe(true);
  });

  test("wraps boundary-state autoregression in a loop zone", () => {
    const recurrentGraph: GraphReport = {
      path: "decoder",
      name: "decoder",
      inputs: ["past_k_layer_0", "iy"],
      outputs: ["present_k_layer_0", "y", "stop_condition_tensor"],
      values: [
        { name: "past_k_layer_0", data_type: "FLOAT", shape: ["n", "1", "4"] },
        { name: "iy", data_type: "INT64", shape: ["1", "n"] },
        { name: "present_k_layer_0", data_type: "FLOAT", shape: ["n", "1", "4"] },
        { name: "y", data_type: "INT64", shape: ["1", "n"] },
        { name: "stop_condition_tensor", data_type: "BOOL", shape: [] },
      ],
      operators: [
        operator(0, "/decoder/step", "Gemm", ["past_k_layer_0", "iy"], ["y"]),
      ],
    };

    const content = buildModelOverview(report([model("stage_decoder.onnx", recurrentGraph)]));
    const module = content.nodes.find((node) => node.data.kind === "module");
    const zone = content.nodes.find((node) => node.data.kind === "loop-zone");
    const feedback = content.edges.find((edge) => edge.type === "feedback");

    expect(zone?.data.loop_state_count).toBe(2);
    expect(zone?.data.loop_condition).toBe("stop_condition_tensor");
    expect(module?.data.is_cyclic).toBe(true);
    expect(module?.data.loop_state_count).toBe(2);
    expect(feedback?.data?.label).toBe("ITERATION STATE × 2");
  });
});
