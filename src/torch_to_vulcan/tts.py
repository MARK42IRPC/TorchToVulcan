"""CPU ONNX TTS inference support for the dedicated listening UI.

This module intentionally favors inspectability over throughput.  TTS exports in
the wild do not share one input contract, so the runtime exposes discovered
inputs and accepts JSON overrides while still providing useful defaults for
common text, token, length, speaker, and sampling controls.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

import numpy as np


class TTSInferenceError(ValueError):
    """A TTS model cannot be loaded or its output cannot become audio."""


@dataclass(frozen=True, slots=True)
class TTSIOInfo:
    name: str
    data_type: str
    shape: tuple[Any, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "shape": [str(item) if item is not None else "?" for item in self.shape],
        }


@dataclass(frozen=True, slots=True)
class TTSModelInfo:
    model_id: str
    name: str
    path: str
    inputs: tuple[TTSIOInfo, ...]
    outputs: tuple[TTSIOInfo, ...]
    sample_rate: int
    source_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path,
            "source_name": self.source_name,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "sample_rate": self.sample_rate,
        }


@dataclass(slots=True)
class _LoadedModel:
    info: TTSModelInfo
    session: Any
    root: Path
    config: dict[str, Any]


_DTYPE_PATTERN = re.compile(r"tensor\(([^)]+)\)", re.IGNORECASE)
_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "bool": np.dtype(np.bool_),
    "float16": np.dtype(np.float16),
    "float": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "float64": np.dtype(np.float64),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "uint64": np.dtype(np.uint64),
    "string": np.dtype(np.str_),
}
_INTEGER_DTYPES = {item.name for item in _DTYPE_MAP.values() if item.kind in "iu"}
_AUDIO_NAME_HINTS = (
    "audio",
    "wave",
    "wav",
    "speech",
    "samples",
    "waveform",
    "pcm",
    "output",
    "y",
)
_TEXT_NAME_HINTS = ("text", "sentence", "prompt", "raw_text", "phoneme")
_TOKEN_NAME_HINTS = ("input_ids", "token", "tokens", "phoneme_ids", "phones", "ids")
_LENGTH_NAME_HINTS = ("length", "lengths", "text_len", "text_length", "seq_len")
_SPEAKER_NAME_HINTS = ("speaker", "spk", "sid", "voice", "speaker_id")
_LANGUAGE_NAME_HINTS = ("lang", "language", "language_id")
_SCALE_NAME_HINTS = ("scale", "length_scale", "speed", "alpha", "noise")


class TTSModelStore:
    """Keep uploaded ONNX TTS sessions alive for the listening WebUI."""

    def __init__(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="torch-to-vulcan-tts-"))
        self._models: dict[str, _LoadedModel] = {}
        self._audio: dict[str, bytes] = {}

    def close(self) -> None:
        self._models.clear()
        self._audio.clear()
        shutil.rmtree(self._root, ignore_errors=True)

    def load_upload(self, filename: str, payload: bytes) -> tuple[TTSModelInfo, ...]:
        if not payload:
            raise TTSInferenceError("TTS 模型文件为空")
        safe_name = Path(filename).name or "tts-model.onnx"
        upload_id = uuid.uuid4().hex
        destination = self._root / upload_id
        destination.mkdir(parents=True, exist_ok=True)
        source = destination / safe_name
        source.write_bytes(payload)
        onnx_paths = self._materialize_models(source, destination)
        if not onnx_paths:
            raise TTSInferenceError("上传包中没有找到 .onnx 模型")
        loaded: list[TTSModelInfo] = []
        for model_path in onnx_paths:
            loaded.append(self._load_model(model_path, safe_name))
        return tuple(loaded)

    def list_models(self) -> tuple[TTSModelInfo, ...]:
        return tuple(item.info for item in self._models.values())

    def get_model(self, model_id: str) -> _LoadedModel:
        try:
            return self._models[model_id]
        except KeyError as error:
            raise TTSInferenceError(f"未知 TTS model_id: {model_id}") from error

    def get_audio(self, audio_id: str) -> bytes:
        try:
            return self._audio[audio_id]
        except KeyError as error:
            raise TTSInferenceError(f"未知音频 ID: {audio_id}") from error

    def synthesize(
        self,
        model_id: str,
        text: str,
        *,
        overrides: Mapping[str, Any] | None = None,
        sample_rate: int | None = None,
    ) -> dict[str, object]:
        if not text.strip():
            raise TTSInferenceError("请输入要合成的文本")
        loaded = self.get_model(model_id)
        feeds = self._prepare_feeds(loaded, text, overrides or {})
        try:
            values = loaded.session.run(None, feeds)
        except Exception as error:
            raise TTSInferenceError(f"ONNX Runtime 推理失败: {error}") from error
        audio, output_info = self._extract_audio(loaded.info.outputs, values)
        rate = int(sample_rate or loaded.info.sample_rate or 22050)
        if rate < 8000 or rate > 192000:
            raise TTSInferenceError(f"采样率 {rate} 不在 8000-192000 Hz 范围内")
        wav = _wav_bytes(audio, rate)
        audio_id = uuid.uuid4().hex
        self._audio[audio_id] = wav
        while len(self._audio) > 16:
            self._audio.pop(next(iter(self._audio)))
        return {
            "audio_id": audio_id,
            "sample_rate": rate,
            "duration_ms": round(len(audio) * 1000 / rate, 2),
            "samples": int(len(audio)),
            "output": output_info.to_dict(),
            "inputs": {
                name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in feeds.items()
            },
        }

    def _materialize_models(self, source: Path, destination: Path) -> list[Path]:
        if source.name.lower().endswith(".onnx"):
            return [source]
        if not source.name.lower().endswith(".zip"):
            raise TTSInferenceError("TTS 推理端目前支持 .onnx 或 .zip")
        extract_root = destination / "extracted"
        extract_root.mkdir()
        try:
            with ZipFile(source) as archive:
                entries = [item for item in archive.infolist() if not item.is_dir()]
                if len(entries) > 512:
                    raise TTSInferenceError("ZIP 文件条目过多")
                for entry in entries:
                    target = (extract_root / entry.filename).resolve()
                    if extract_root.resolve() not in target.parents:
                        raise TTSInferenceError("ZIP 包含越界路径")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(entry))
        except (BadZipFile, OSError, ValueError) as error:
            raise TTSInferenceError(f"无法读取 TTS ZIP: {error}") from error
        return sorted(extract_root.rglob("*.onnx"), key=lambda item: item.as_posix().casefold())

    def _load_model(self, path: Path, source_name: str) -> TTSModelInfo:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise TTSInferenceError("TTS 推理需要 onnxruntime，请运行安装脚本") from error
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as error:
            raise TTSInferenceError(f"无法加载 ONNX TTS 模型 {path.name}: {error}") from error
        inputs = tuple(_io_info(item) for item in session.get_inputs())
        outputs = tuple(_io_info(item) for item in session.get_outputs())
        config = _load_config(path.parent)
        model_id = uuid.uuid4().hex
        info = TTSModelInfo(
            model_id=model_id,
            name=path.stem,
            path=path.name,
            inputs=inputs,
            outputs=outputs,
            sample_rate=_sample_rate(config),
            source_name=source_name,
        )
        self._models[model_id] = _LoadedModel(info, session, path.parent, config)
        return info

    def _prepare_feeds(
        self,
        loaded: _LoadedModel,
        text: str,
        overrides: Mapping[str, Any],
    ) -> dict[str, np.ndarray[Any, Any]]:
        feeds: dict[str, np.ndarray[Any, Any]] = {}
        for item in loaded.info.inputs:
            if item.name in overrides:
                feeds[item.name] = _coerce_override(overrides[item.name], item)
                continue
            dtype = _numpy_dtype(item.data_type)
            lower = item.name.casefold()
            if dtype.kind == "U":
                value: Any = text
            elif any(hint in lower for hint in _TOKEN_NAME_HINTS):
                value = _token_ids(text, loaded.config)
            elif any(hint in lower for hint in _LENGTH_NAME_HINTS):
                value = len(_token_ids(text, loaded.config))
            elif any(hint in lower for hint in _SPEAKER_NAME_HINTS):
                value = 0
            elif any(hint in lower for hint in _LANGUAGE_NAME_HINTS):
                value = 0
            elif any(hint in lower for hint in _SCALE_NAME_HINTS):
                value = 1.0
            elif dtype.kind in "iu":
                value = 0
            elif dtype.kind == "b":
                value = False
            else:
                value = 0.0
            feeds[item.name] = _fit_value(value, item, dtype)
        return feeds

    def _extract_audio(
        self,
        infos: tuple[TTSIOInfo, ...],
        values: list[Any],
    ) -> tuple[np.ndarray[Any, Any], TTSIOInfo]:
        candidates: list[tuple[int, TTSIOInfo, np.ndarray[Any, Any]]] = []
        mel_outputs: list[TTSIOInfo] = []
        for info, raw in zip(infos, values, strict=False):
            array = np.asarray(raw)
            if array.dtype.kind not in "fiu" or array.size < 16:
                continue
            name = info.name.casefold()
            if _looks_like_mel(info, array):
                mel_outputs.append(info)
                continue
            if not _is_direct_waveform_shape(array.shape):
                continue
            score = 0
            if any(hint in name for hint in _AUDIO_NAME_HINTS):
                score += 20
            if array.ndim == 1:
                score += 20
            elif array.ndim == 2:
                score += 15
            else:
                # [batch, channel, samples] is accepted only for a singleton
                # batch and channel, then squeezed below.
                score += 10
            candidates.append((score + min(array.size, 1_000_000) // 1_000_000, info, array))
        if not candidates:
            names = ", ".join(item.name for item in infos) or "(无输出)"
            if mel_outputs:
                mel_names = ", ".join(item.name for item in mel_outputs)
                raise TTSInferenceError(
                    f"模型输出 {mel_names} 是 Mel 频谱，不是可直接播放的 waveform。"
                    "请同时上传 vocoder，或在导出时选择 waveform 音频输出。"
                )
            raise TTSInferenceError(
                f"模型没有发现可直接播放的一维音频输出；输出为 {names}。"
                "如果这是 Mel 频谱模型，请同时准备 vocoder。"
            )
        _, info, array = max(candidates, key=lambda item: item[0])
        audio = np.asarray(_squeeze_waveform(array), dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 1e-8:
            raise TTSInferenceError("模型输出音频全为 0，无法生成可听 WAV；请检查输入覆盖值")
        if peak > 1.0:
            audio = audio / peak
        return np.clip(audio, -1.0, 1.0), info


def _io_info(value: Any) -> TTSIOInfo:
    shape = tuple(getattr(value, "shape", ()) or ())
    return TTSIOInfo(str(value.name), str(value.type), shape)


def _numpy_dtype(data_type: str) -> np.dtype[Any]:
    match = _DTYPE_PATTERN.search(data_type)
    key = (match.group(1) if match else data_type).casefold()
    return _DTYPE_MAP.get(key, np.dtype(np.float32))


def _fit_value(value: Any, info: TTSIOInfo, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    shape = info.shape
    if dtype.kind == "U":
        array = np.asarray(value, dtype=np.str_)
    else:
        array = np.asarray(value, dtype=dtype)
    if not shape:
        return np.asarray(array.reshape(()), dtype=dtype)
    expected = [int(item) if isinstance(item, int) and item > 0 else None for item in shape]
    if array.ndim == 0:
        array = array.reshape((1,) * len(expected))
    if len(expected) == 2 and array.ndim == 1 and expected[0] in (None, 1):
        array = array.reshape((1, -1))
    if len(expected) == 1 and array.ndim > 1:
        array = array.reshape(-1)
    if expected and expected[-1] is not None and array.shape[-1] != expected[-1]:
        size = expected[-1]
        if array.shape[-1] > size:
            array = array[..., :size]
        else:
            pad_shape = (*array.shape[:-1], size - array.shape[-1])
            array = np.concatenate((array, np.zeros(pad_shape, dtype=dtype)), axis=-1)
    return np.ascontiguousarray(array, dtype=dtype)


def _coerce_override(value: Any, info: TTSIOInfo) -> np.ndarray[Any, Any]:
    if isinstance(value, str) and _numpy_dtype(info.data_type).kind == "U":
        return np.asarray(value, dtype=np.str_)
    return _fit_value(value, info, _numpy_dtype(info.data_type))


def _is_direct_waveform_shape(shape: tuple[int, ...]) -> bool:
    """Return whether a tensor can be interpreted as mono waveform samples."""
    if len(shape) == 1:
        return shape[0] >= 2
    if len(shape) == 2:
        return shape[0] == 1 or shape[1] == 1
    if len(shape) == 3:
        return shape[0] == 1 and shape[1] == 1 and shape[2] >= 2
    return False


def _squeeze_waveform(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Remove the batch/channel singleton dimensions accepted above."""
    squeezed = np.squeeze(array)
    if squeezed.ndim != 1:
        raise TTSInferenceError(f"waveform 输出形状 {list(array.shape)} 不是单声道一维音频")
    return squeezed


def _looks_like_mel(info: TTSIOInfo, array: np.ndarray[Any, Any]) -> bool:
    """Recognize common mel/spectrogram outputs before any flattening occurs."""
    name = info.name.casefold()
    if any(token in name for token in ("mel", "spectrogram", "spec", "stft")):
        return array.ndim >= 2
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[1] in {40, 64, 80, 100}:
        return True
    if array.ndim == 2 and min(array.shape) in {40, 64, 80, 100} and max(array.shape) > min(array.shape):
        return True
    return False


def _token_ids(text: str, config: Mapping[str, Any]) -> list[int]:
    vocab = _find_vocab(config)
    if vocab:
        unknown = _lookup(vocab, ("<unk>", "[UNK]", "unk"), 0)
        ids = [int(vocab.get(char, unknown)) for char in text]
        bos = _lookup(vocab, ("<bos>", "[BOS]", "<s>"), None)
        eos = _lookup(vocab, ("<eos>", "[EOS]", "</s>"), None)
        return ([bos] if bos is not None else []) + ids + ([eos] if eos is not None else [])
    # A deterministic fallback lets a graph be smoke-tested before its real
    # tokenizer is supplied. It is not intended to replace model vocabulary.
    vocab_size = int(config.get("vocab_size", 1024) or 1024)
    return [1 + (ord(char) % max(2, vocab_size - 1)) for char in text]


def _find_vocab(config: Mapping[str, Any]) -> dict[str, int]:
    for key in ("vocab", "vocabulary", "token_to_id", "phoneme_id_map"):
        value = config.get(key)
        if isinstance(value, dict):
            result = {str(item): int(index) for item, index in value.items() if isinstance(index, (int, float))}
            if result:
                return result
    return {}


def _lookup(vocab: Mapping[str, int], names: tuple[str, ...], default: int | None) -> int | None:
    for name in names:
        if name in vocab:
            return vocab[name]
    return default


def _load_config(directory: Path) -> dict[str, Any]:
    for name in ("config.json", "config.yaml", "model.json", "metadata.json"):
        path = directory / name
        if path.suffix != ".json" or not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _sample_rate(config: Mapping[str, Any]) -> int:
    for key in ("sample_rate", "sampling_rate", "sr"):
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    for key in ("audio", "model", "vocoder"):
        nested = config.get(key)
        if isinstance(nested, dict):
            result = _sample_rate(nested)
            if result:
                return result
    return 22050


def _wav_bytes(audio: np.ndarray[Any, Any], sample_rate: int) -> bytes:
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype=np.int16)
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    return stream.getvalue()
