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
from fastapi.responses import StreamingResponse

from .importer import (
    InspectionError,
    MemoryConfirmationRequired,
    inspect_path,
    source_format,
    supported_input_suffixes,
)


UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Torch to Vulcan API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "api_version": "0.3",
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
