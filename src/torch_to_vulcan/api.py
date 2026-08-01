"""HTTP adapter for the ONNX inspection service."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from .compiler.vulkan import VerificationRunner, VerificationTarget
from .importer import (
    InspectionError,
    MemoryConfirmationRequired,
    inspect_path,
    source_format,
    supported_input_suffixes,
)
from .tts import TTSInferenceError, TTSModelStore


UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Torch to Vulcan API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
tts_store = TTSModelStore()


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "api_version": "0.4",
        "formats": list(supported_input_suffixes()),
    }


@app.post("/api/inspect")
async def inspect_upload(
    file: Annotated[UploadFile, File(...)],
    confirm_large_model: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    filename = Path(file.filename or "upload").name
    if source_format(filename) is None:
        expected = ", ".join(supported_input_suffixes())
        raise HTTPException(status_code=400, detail=f"unsupported file type; expected {expected}")

    with TemporaryDirectory(prefix="torch-to-vulcan-") as temporary_directory:
        upload_path = Path(temporary_directory) / filename
        try:
            with upload_path.open("wb") as stream:
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    stream.write(chunk)

            report = await run_in_threadpool(
                inspect_path,
                upload_path,
                confirm_large_model=confirm_large_model,
            )
        except MemoryConfirmationRequired as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "memory_confirmation_required",
                    "model_path": error.model_path,
                    "estimated_bytes": error.estimated_bytes,
                    "available_bytes": error.available_bytes,
                    "threshold_bytes": int(error.available_bytes * error.warning_ratio),
                    "warning_ratio": error.warning_ratio,
                },
            ) from error
        except InspectionError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            await file.close()

    result = report.to_dict()
    result["source"] = filename
    return result


@app.post("/api/inspect/stream")
async def inspect_upload_stream(
    file: Annotated[UploadFile, File(...)],
    confirm_large_model: Annotated[bool, Form()] = False,
) -> StreamingResponse:
    filename = Path(file.filename or "upload").name
    if source_format(filename) is None:
        expected = ", ".join(supported_input_suffixes())
        raise HTTPException(status_code=400, detail=f"unsupported file type; expected {expected}")

    temporary_directory = TemporaryDirectory(prefix="torch-to-vulcan-")
    upload_path = Path(temporary_directory.name) / filename
    try:
        with upload_path.open("wb") as stream:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                stream.write(chunk)
    except Exception:
        temporary_directory.cleanup()
        raise
    finally:
        await file.close()

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        def publish(event: dict[str, object]) -> None:
            loop.call_soon_threadsafe(
                events.put_nowait,
                {"type": "progress", "progress": event},
            )

        async def inspect() -> None:
            try:
                report = await run_in_threadpool(
                    inspect_path,
                    upload_path,
                    confirm_large_model=confirm_large_model,
                    progress=publish,
                )
                result = report.to_dict()
                result["source"] = filename
                await events.put({"type": "result", "report": result})
            except MemoryConfirmationRequired as error:
                await events.put(
                    {
                        "type": "memory_warning",
                        "warning": {
                            "code": "memory_confirmation_required",
                            "model_path": error.model_path,
                            "estimated_bytes": error.estimated_bytes,
                            "available_bytes": error.available_bytes,
                            "threshold_bytes": int(
                                error.available_bytes * error.warning_ratio
                            ),
                            "warning_ratio": error.warning_ratio,
                        },
                    }
                )
            except InspectionError as error:
                await events.put({"type": "error", "message": str(error)})
            except Exception as error:
                await events.put(
                    {"type": "error", "message": str(error) or type(error).__name__}
                )

        task = asyncio.create_task(inspect())
        try:
            while True:
                event = await events.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"result", "memory_warning", "error"}:
                    break
            await task
        finally:
            if not task.done():
                task.cancel()
            temporary_directory.cleanup()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/verify/stream")
async def verify_mappings_stream(payload: dict[str, Any]) -> StreamingResponse:
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise HTTPException(status_code=400, detail="targets must be an array")
    try:
        targets = tuple(VerificationTarget.from_mapping(item) for item in raw_targets)
    except (AttributeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"invalid verification target: {error}",
        ) from error

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        def publish(event: dict[str, object]) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        async def verify() -> None:
            try:
                runner = VerificationRunner()
                await run_in_threadpool(runner.run, targets, publish)
            except Exception as error:
                await events.put(
                    {"type": "error", "message": str(error) or type(error).__name__}
                )

        task = asyncio.create_task(verify())
        try:
            while True:
                event = await events.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"result", "error"}:
                    break
            await task
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/api/tts/models")
def list_tts_models() -> dict[str, object]:
    """List models currently loaded by the dedicated listening UI."""
    return {"models": [item.to_dict() for item in tts_store.list_models()]}


@app.post("/api/tts/models")
async def upload_tts_model(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    filename = Path(file.filename or "tts-model.onnx").name
    if not filename.lower().endswith((".onnx", ".zip")):
        raise HTTPException(status_code=400, detail="TTS 推理端目前支持 .onnx 或 .zip")
    try:
        payload = await file.read()
        models = await run_in_threadpool(tts_store.load_upload, filename, payload)
    except TTSInferenceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        await file.close()
    return {"models": [item.to_dict() for item in models]}


@app.post("/api/tts/synthesize")
async def synthesize_tts(
    model_id: Annotated[str, Form(...)],
    text: Annotated[str, Form(...)],
    overrides: Annotated[str, Form()] = "{}",
    sample_rate: Annotated[int | None, Form()] = None,
) -> dict[str, object]:
    try:
        parsed_overrides = json.loads(overrides or "{}")
        if not isinstance(parsed_overrides, dict):
            raise ValueError("overrides 必须是 JSON object")
        result = await run_in_threadpool(
            tts_store.synthesize,
            model_id,
            text,
            overrides=parsed_overrides,
            sample_rate=sample_rate,
        )
    except (TTSInferenceError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    result["audio_url"] = f"/api/tts/audio/{result['audio_id']}"
    return result


@app.get("/api/tts/audio/{audio_id}")
def get_tts_audio(audio_id: str) -> Response:
    try:
        payload = tts_store.get_audio(audio_id)
    except TTSInferenceError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="torch-to-vulcan-{audio_id}.wav"'},
    )
