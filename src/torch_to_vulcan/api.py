"""HTTP adapter for the ONNX inspection service."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .importer import InspectionError, inspect_path, source_format, supported_input_suffixes


UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

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
        "api_version": "0.2",
        "formats": list(supported_input_suffixes()),
    }


@app.post("/api/inspect")
async def inspect_upload(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    filename = Path(file.filename or "upload").name
    if source_format(filename) is None:
        expected = ", ".join(supported_input_suffixes())
        raise HTTPException(status_code=400, detail=f"unsupported file type; expected {expected}")

    with TemporaryDirectory(prefix="torch-to-vulcan-") as temporary_directory:
        upload_path = Path(temporary_directory) / filename
        size = 0
        try:
            with upload_path.open("wb") as stream:
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="upload exceeds 1 GiB limit")
                    stream.write(chunk)

            report = await run_in_threadpool(inspect_path, upload_path)
        except InspectionError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            await file.close()

    result = report.to_dict()
    result["source"] = filename
    return result
