from __future__ import annotations

import io
import json
import unittest
import wave
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from torch_to_vulcan.api import app, tts_store
from torch_to_vulcan.tts import TTSInferenceError, TTSIOInfo, TTSModelStore


def make_wave_model() -> bytes:
    output = helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, 64])
    samples = np.sin(np.linspace(0, 3.14, 64, dtype=np.float32)).reshape(1, 64)
    constant = numpy_helper.from_array(samples, name="samples")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["waveform"], value=constant)],
        "wave_fixture",
        [],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()


def make_input_contract_model() -> bytes:
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "sequence"])
    input_lengths = helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1])
    output = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, 64])
    samples = np.sin(np.linspace(0, 3.14, 64, dtype=np.float32)).reshape(1, 1, 64)
    constant = numpy_helper.from_array(samples, name="samples")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["audio"], value=constant)],
        "input_contract_fixture",
        [input_ids, input_lengths],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()


def make_mel_model() -> bytes:
    output = helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, 80, 10])
    constant = numpy_helper.from_array(np.ones((1, 80, 10), dtype=np.float32), name="mel_values")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["mel"], value=constant)],
        "mel_fixture",
        [],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()


class TTSApiTests(unittest.TestCase):
    client = TestClient(app)

    def tearDown(self) -> None:
        tts_store._models.clear()
        tts_store._audio.clear()

    def test_upload_synthesize_and_read_wav(self) -> None:
        response = self.client.post(
            "/api/tts/models",
            files={"file": ("fixture.onnx", io.BytesIO(make_wave_model()), "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 200)
        model = response.json()["models"][0]
        self.assertEqual(model["outputs"][0]["name"], "waveform")

        result = self.client.post(
            "/api/tts/synthesize",
            data={"model_id": model["model_id"], "text": "hello"},
        )
        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(payload["samples"], 64)

        audio = self.client.get(payload["audio_url"])
        self.assertEqual(audio.status_code, 200)
        with wave.open(io.BytesIO(audio.content), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getframerate(), 22050)
            self.assertEqual(reader.getnframes(), 64)

    def test_zip_upload_and_invalid_override(self) -> None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as writer:
            writer.writestr("nested/fixture.onnx", make_wave_model())
        response = self.client.post(
            "/api/tts/models",
            files={"file": ("fixture.zip", io.BytesIO(archive.getvalue()), "application/zip")},
        )
        self.assertEqual(response.status_code, 200)
        model_id = response.json()["models"][0]["model_id"]
        result = self.client.post(
            "/api/tts/synthesize",
            data={"model_id": model_id, "text": "hello", "overrides": "[]"},
        )
        self.assertEqual(result.status_code, 400)
        self.assertIn("JSON object", result.json()["detail"])

    def test_dynamic_token_and_length_inputs_keep_expected_shapes(self) -> None:
        response = self.client.post(
            "/api/tts/models",
            files={"file": ("contract.onnx", io.BytesIO(make_input_contract_model()), "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 200)
        model_id = response.json()["models"][0]["model_id"]
        result = self.client.post(
            "/api/tts/synthesize",
            data={"model_id": model_id, "text": "hello"},
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["inputs"]["input_ids"]["shape"][0], 1)
        self.assertEqual(result.json()["inputs"]["input_lengths"]["shape"], [1])
        self.assertEqual(result.json()["samples"], 64)

    def test_mel_output_is_rejected_instead_of_flattened(self) -> None:
        response = self.client.post(
            "/api/tts/models",
            files={"file": ("mel.onnx", io.BytesIO(make_mel_model()), "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 200)
        model_id = response.json()["models"][0]["model_id"]
        result = self.client.post(
            "/api/tts/synthesize",
            data={"model_id": model_id, "text": "hello"},
        )
        self.assertEqual(result.status_code, 400)
        self.assertIn("Mel", result.json()["detail"])

    def test_waveform_shape_helpers_accept_mono_forms_only(self) -> None:
        store = TTSModelStore()
        try:
            for shape in ((64,), (1, 64), (64, 1), (1, 1, 64)):
                audio, _ = store._extract_audio(
                    (TTSIOInfo("audio", "tensor(float)", shape),),
                    [np.ones(shape, dtype=np.float32)],
                )
                self.assertEqual(audio.shape, (64,))
            with self.assertRaises(TTSInferenceError):
                store._extract_audio(
                    (TTSIOInfo("audio", "tensor(float)", (2, 32)),),
                    [np.ones((2, 32), dtype=np.float32)],
                )
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
