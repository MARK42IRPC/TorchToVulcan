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
      operator_count: 6,
      graphs: [
        {
          path: "vision_classifier",
          name: "vision_classifier",
          inputs: ["image"],
          outputs: ["probabilities"],
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
              name: "features/pool",
              op_type: "GlobalAveragePool",
              domain: "",
              inputs: ["features_1"],
              outputs: ["pooled"],
            },
            {
              graph_path: "vision_classifier",
              index: 3,
              name: "classifier/flatten",
              op_type: "Flatten",
              domain: "",
              inputs: ["pooled"],
              outputs: ["flat"],
            },
            {
              graph_path: "vision_classifier",
              index: 4,
              name: "classifier/gemm",
              op_type: "Gemm",
              domain: "",
              inputs: ["flat", "fc.weight", "fc.bias"],
              outputs: ["logits"],
            },
            {
              graph_path: "vision_classifier",
              index: 5,
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
  operator_count: 6,
  operator_summary: [
    { domain: "", op_type: "Conv", count: 1 },
    { domain: "", op_type: "Flatten", count: 1 },
    { domain: "", op_type: "Gemm", count: 1 },
    { domain: "", op_type: "GlobalAveragePool", count: 1 },
    { domain: "", op_type: "Relu", count: 1 },
    { domain: "", op_type: "Softmax", count: 1 },
  ],
};

async function loadMockModel(page: Page) {
  await page.route("**/api/inspect", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(mockReport) });
  });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "vision_classifier.onnx",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("mock"),
  });
  await expect(page.getByText("vision_classifier.onnx").first()).toBeVisible();
  await expect(page.locator(".graph-node")).toHaveCount(8);
}

test("renders and inspects a loaded graph on desktop", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await loadMockModel(page);

  await page.getByRole("button", { name: /Relu/ }).click();
  await expect(page.locator(".node-detail__title > strong")).toHaveText("Relu");
  await expect(page.locator(".property-list").getByText("PENDING")).toBeVisible();

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: "test-results/webui-desktop.png", fullPage: true });
});

test("keeps controls within the mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await loadMockModel(page);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: "test-results/webui-mobile.png", fullPage: true });
});
