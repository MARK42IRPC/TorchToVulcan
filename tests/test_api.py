from __future__ import annotations

import gzip
import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

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
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["api_version"], "0.3")
        self.assertIn(".7z", response.json()["formats"])
        self.assertIn(".rar", response.json()["formats"])

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
        tensor_values = {
            item["name"]: item for item in value["models"][0]["graphs"][0]["values"]
        }
        self.assertEqual(tensor_values["input"]["data_type"], "FLOAT")
        self.assertEqual(tensor_values["input"]["shape"], ["1", "4"])

    def test_rejects_unsupported_file_type(self) -> None:
        response = self.client.post(
            "/api/inspect",
            files={"file": ("model.pth", BytesIO(b"future"), "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(".7z", response.json()["detail"])

    def test_inspects_uploaded_gzip_onnx(self) -> None:
        response = self.client.post(
            "/api/inspect",
            files={
                "file": (
                    "relu.onnx.gz",
                    BytesIO(gzip.compress(make_relu_model())),
                    "application/gzip",
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_type"], "gzip")
        self.assertEqual(response.json()["operator_summary"][0]["op_type"], "Relu")

    def test_large_model_requires_explicit_confirmation(self) -> None:
        with patch(
            "torch_to_vulcan.importer.inspector.psutil.virtual_memory",
            return_value=SimpleNamespace(available=1),
        ):
            warning = self.client.post(
                "/api/inspect",
                files={
                    "file": (
                        "large.onnx",
                        BytesIO(make_relu_model()),
                        "application/octet-stream",
                    )
                },
            )
            confirmed = self.client.post(
                "/api/inspect",
                data={"confirm_large_model": "true"},
                files={
                    "file": (
                        "large.onnx",
                        BytesIO(make_relu_model()),
                        "application/octet-stream",
                    )
                },
            )

        self.assertEqual(warning.status_code, 409)
        self.assertEqual(warning.json()["detail"]["code"], "memory_confirmation_required")
        self.assertEqual(warning.json()["detail"]["warning_ratio"], 0.6)
        self.assertEqual(confirmed.status_code, 200)

    def test_streams_import_progress_and_result(self) -> None:
        response = self.client.post(
            "/api/inspect/stream",
            files={"file": ("relu.onnx", BytesIO(make_relu_model()), "application/octet-stream")},
        )

        events = [json.loads(line) for line in response.text.splitlines()]
        progress = [event["progress"] for event in events if event["type"] == "progress"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(progress[0]["phase"], "scanning")
        self.assertEqual(progress[-1]["percent"], 100.0)
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["report"]["operator_count"], 1)


if __name__ == "__main__":
    unittest.main()
