import { expect, test, type Page } from "@playwright/test";

const mockReport = {
  source: "vision_classifier.onnx",
  source_type: "onnx",
  models: [
    {
      path: "vision_classifier.onnx",
      graph_name: "vision_classifier",
      ir_version: 10,
      producer_name: "torch",
      producer_version: "2.7",
      opsets: [{ domain: "", version: 18 }],
      operator_count: 7,
      graphs: [
        {
          path: "vision_classifier",
          name: "vision_classifier",
          inputs: ["image"],
          outputs: ["probabilities"],
          values: [
            { name: "image", data_type: "FLOAT", shape: ["1", "3", "224", "224"] },
            { name: "conv.weight", data_type: "FLOAT", shape: ["32", "3", "3", "3"] },
            { name: "conv.bias", data_type: "FLOAT", shape: ["32"] },
            { name: "features_0", data_type: "FLOAT", shape: ["1", "32", "112", "112"] },
            { name: "features_1", data_type: "FLOAT", shape: ["1", "32", "112", "112"] },
            { name: "features_2", data_type: "FLOAT", shape: ["1", "32", "112", "112"] },
            { name: "pooled", data_type: "FLOAT", shape: ["1", "32", "1", "1"] },
            { name: "flat", data_type: "FLOAT", shape: ["1", "32"] },
            { name: "fc.weight", data_type: "FLOAT", shape: ["1000", "32"] },
            { name: "fc.bias", data_type: "FLOAT", shape: ["1000"] },
            { name: "logits", data_type: "FLOAT", shape: ["1", "1000"] },
            { name: "probabilities", data_type: "FLOAT", shape: ["1", "1000"] },
          ],
          operators: [
            {
              graph_path: "vision_classifier",
              index: 0,
              name: "features/conv",
              op_type: "Conv",
              domain: "",
              inputs: ["image", "conv.weight", "conv.bias"],
              outputs: ["features_0"],
            },
            {
              graph_path: "vision_classifier",
              index: 1,
              name: "features/relu",
              op_type: "Relu",
              domain: "",
              inputs: ["features_0"],
              outputs: ["features_1"],
            },
            {
              graph_path: "vision_classifier",
              index: 2,
              name: "features/relu_2",
              op_type: "Relu",
              domain: "",
              inputs: ["features_1"],
              outputs: ["features_2"],
            },
            {
              graph_path: "vision_classifier",
              index: 3,
              name: "features/pool",
              op_type: "GlobalAveragePool",
              domain: "",
              inputs: ["features_2"],
              outputs: ["pooled"],
            },
            {
              graph_path: "vision_classifier",
              index: 4,
              name: "classifier/flatten",
              op_type: "Flatten",
              domain: "",
              inputs: ["pooled"],
              outputs: ["flat"],
            },
            {
              graph_path: "vision_classifier",
              index: 5,
              name: "classifier/gemm",
              op_type: "Gemm",
              domain: "",
              inputs: ["flat", "fc.weight", "fc.bias"],
              outputs: ["logits"],
            },
            {
              graph_path: "vision_classifier",
              index: 6,
              name: "classifier/softmax",
              op_type: "Softmax",
              domain: "",
              inputs: ["logits"],
              outputs: ["probabilities"],
            },
          ],
        },
      ],
    },
  ],
  errors: [],
  operator_count: 7,
  operator_summary: [
    { domain: "", op_type: "Conv", count: 1 },
    { domain: "", op_type: "Flatten", count: 1 },
    { domain: "", op_type: "Gemm", count: 1 },
    { domain: "", op_type: "GlobalAveragePool", count: 1 },
    { domain: "", op_type: "Relu", count: 2 },
    { domain: "", op_type: "Softmax", count: 1 },
  ],
};

async function loadMockModel(page: Page) {
  await page.route("**/api/inspect/stream", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/x-ndjson",
      body: `${JSON.stringify({ type: "result", report: mockReport })}\n`,
    });
  });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "vision_classifier.onnx",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
  await expect(page.getByText("vision_classifier.onnx").first()).toBeVisible();
  await expect(page.locator(".graph-node--module")).toHaveCount(1);
}

test("renders and inspects a loaded graph on desktop", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await loadMockModel(page);

  await page.locator(".graph-node--module").dblclick();
  await expect(page.locator(".graph-node--group")).toHaveCount(2);
  await page.locator(".graph-node--group", { hasText: "features" }).dblclick();
  await expect(page.locator(".graph-node--operator")).toHaveCount(4);

  await page.getByRole("button", { name: /Relu/ }).click();
  await expect(page.locator(".operator-list button", { hasText: "Relu" })).toHaveCount(1);
  const reluGroup = page.locator(".operator-list button", { hasText: "Relu" });
  await expect(reluGroup.locator(".operator-count")).toHaveText("2");
  await expect(page.locator(".node-detail__title > strong")).toHaveText("Relu");
  await expect(page.locator(".property-list").getByText("PENDING")).toBeVisible();
  await expect(page.locator(".tensor-list").first().getByText("FLOAT")).toBeVisible();
  await expect(
    page.locator(".tensor-list").first().getByText("[1 × 32 × 112 × 112]"),
  ).toBeVisible();

  const nodeX = async (label: string) => {
    const box = await page.locator(".graph-node")
      .filter({ has: page.getByText(label, { exact: true }) })
      .first()
      .boundingBox();
    if (!box) throw new Error(`missing graph node: ${label}`);
    return box.x;
  };
  expect(await nodeX("image")).toBeLessThan(await nodeX("Conv"));
  expect(await nodeX("Conv")).toBeLessThan(await nodeX("Relu"));
  expect(await nodeX("Relu")).toBeLessThan(await nodeX("GlobalAveragePool"));
  const inputColor = await page.locator(".graph-node--input").first().evaluate(
    (node) => getComputedStyle(node).borderColor,
  );
  expect(inputColor).toBe("rgb(57, 122, 85)");

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({ path: "test-results/webui-desktop.png", fullPage: true });
});

test("keeps controls within the mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await loadMockModel(page);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({ path: "test-results/webui-mobile.png", fullPage: true });
});

test("asks before loading a model above the memory threshold", async ({ page }) => {
  let requestCount = 0;
  await page.route("**/api/inspect/stream", async (route) => {
    requestCount += 1;
    if (requestCount === 1) {
      await route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: `${JSON.stringify({
          type: "memory_warning",
          warning: {
            code: "memory_confirmation_required",
            model_path: "huge.onnx",
            estimated_bytes: 8 * 1024 * 1024 * 1024,
            available_bytes: 10 * 1024 * 1024 * 1024,
            threshold_bytes: 6 * 1024 * 1024 * 1024,
            warning_ratio: 0.6,
          },
        })}\n`,
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/x-ndjson",
      body: `${JSON.stringify({ type: "result", report: mockReport })}\n`,
    });
  });

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "huge.onnx",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
  await expect(page.getByRole("alertdialog")).toBeVisible();
  await expect(page.getByText("模型过大，还要加载吗？")).toBeVisible();
  await page.screenshot({ path: "test-results/memory-warning.png", fullPage: true });
  await page.getByRole("button", { name: "我知道我在做什么" }).click();
  await expect(page.getByRole("alertdialog")).toBeHidden();
  await expect(page.getByText("vision_classifier.onnx").first()).toBeVisible();
  expect(requestCount).toBe(2);
});
