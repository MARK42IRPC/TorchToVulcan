from __future__ import annotations

import unittest
from io import BytesIO

from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from torch_to_vulcan.api import app


def make_relu_model() -> bytes:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["input"], ["output"], name="relu_0")
    graph = helper.make_graph([node], "relu_graph", [input_info], [output_info])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    ).SerializeToString()


class ApiTests(unittest.TestCase):
    client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_inspects_uploaded_onnx(self) -> None:
        response = self.client.post(
            "/api/inspect",
            files={"file": ("relu.onnx", BytesIO(make_relu_model()), "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertEqual(value["source"], "relu.onnx")
        self.assertEqual(value["source_type"], "onnx")
        self.assertEqual(value["operator_summary"][0]["op_type"], "Relu")
        self.assertEqual(value["models"][0]["graphs"][0]["inputs"], ["input"])

    def test_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/api/inspect",
            files={"file": ("model.pth", BytesIO(b"future"), "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("expected .onnx, .zip", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
