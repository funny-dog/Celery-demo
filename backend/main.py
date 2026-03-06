from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import redis.asyncio as redis
from celery.result import AsyncResult
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from api.v1 import auth, users, datasources, tasks, audit, discovery, masking
from config import settings
from database import get_session, init_db
from models import DataRecord
from schemas import (
    SplitUploadChunkResponse,
    SplitUploadCompleteRequest,
    SplitUploadInitRequest,
    SplitUploadInitResponse,
    TaskCancelResponse,
    TaskStatusResponse,
    UploadResponse,
)
from worker import process_csv, process_desensitize, process_split_archive
from celery_app import celery_app

app = FastAPI(title="Bulk Desensitizer")

# Register API v1 routes
app.include_router(auth.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(datasources.router, prefix="/api/v1")
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(discovery.router, prefix="/api/v1")
app.include_router(masking.router, prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dist_dir = Path(settings.frontend_dist_dir)
frontend_assets_dir = frontend_dist_dir / "assets"
if frontend_assets_dir.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(frontend_assets_dir)),
        name="frontend-assets",
    )


def _detect_media_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if suffix == ".zip":
        return "application/zip"
    if suffix == ".txt":
        return "text/plain"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _split_upload_meta_path(upload_dir: Path, upload_id: str) -> Path:
    return upload_dir / f"{upload_id}_split_meta.json"


def _split_upload_temp_path(upload_dir: Path, upload_id: str, suffix: str) -> Path:
    return upload_dir / f"{upload_id}_split_upload{suffix}"


def _safe_upload_id(upload_id: str) -> str:
    normalized = upload_id.strip()
    if not normalized or any(char in normalized for char in {"/", "\\", ".."}):
        raise HTTPException(status_code=400, detail="invalid upload_id")
    return normalized


def _read_split_meta(upload_dir: Path, upload_id: str) -> dict:
    meta_path = _split_upload_meta_path(upload_dir, upload_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="split upload session not found")
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="split upload metadata corrupted") from exc


def _write_split_meta(upload_dir: Path, upload_id: str, meta: dict) -> None:
    meta_path = _split_upload_meta_path(upload_dir, upload_id)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=400, detail="only .csv or .xlsx is supported")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}_{file.filename}"

    with destination.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    task = process_csv.delay(str(destination))
    return UploadResponse(task_id=task.id)


@app.get("/", include_in_schema=False)
def frontend_index():
    index_path = Path(settings.frontend_dist_dir) / "index.html"
    if index_path.exists():
        return FileResponse(path=index_path, media_type="text/html")
    return {"message": "Bulk Desensitizer API is running"}


@app.post("/upload/desensitize", response_model=UploadResponse)
async def upload_desensitize(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=400, detail="only .csv or .xlsx is supported")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}_{file.filename}"

    with destination.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    task = process_desensitize.delay(str(destination))
    return UploadResponse(task_id=task.id)


@app.post("/upload/split", response_model=UploadResponse)
async def upload_split(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="only .pdf or .txt is supported")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}_{file.filename}"

    with destination.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    task = process_split_archive.delay(str(destination), 140)
    return UploadResponse(task_id=task.id)


@app.post("/upload/split/init", response_model=SplitUploadInitResponse)
async def upload_split_init(payload: SplitUploadInitRequest) -> SplitUploadInitResponse:
    filename = payload.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="missing filename")

    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="only .pdf or .txt is supported")

    upload_id = uuid4().hex
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    sanitized_name = Path(filename).name
    meta = {
        "filename": sanitized_name,
        "suffix": suffix,
        "next_chunk_index": 0,
        "total_chunks": None,
    }
    _write_split_meta(upload_dir, upload_id, meta)

    temp_path = _split_upload_temp_path(upload_dir, upload_id, suffix)
    temp_path.write_bytes(b"")

    return SplitUploadInitResponse(upload_id=upload_id)


@app.post("/upload/split/chunk", response_model=SplitUploadChunkResponse)
async def upload_split_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
) -> SplitUploadChunkResponse:
    normalized_upload_id = _safe_upload_id(upload_id)
    if chunk_index < 0:
        raise HTTPException(status_code=400, detail="chunk_index must be >= 0")
    if total_chunks <= 0:
        raise HTTPException(status_code=400, detail="total_chunks must be > 0")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_split_meta(upload_dir, normalized_upload_id)

    expected_index = int(meta.get("next_chunk_index", 0))
    if chunk_index != expected_index:
        raise HTTPException(
            status_code=409,
            detail=f"unexpected chunk index: expected {expected_index}, got {chunk_index}",
        )

    existing_total = meta.get("total_chunks")
    if existing_total is None:
        meta["total_chunks"] = total_chunks
    elif int(existing_total) != total_chunks:
        raise HTTPException(status_code=400, detail="total_chunks mismatch")

    suffix = str(meta.get("suffix", "")).lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=500, detail="invalid split upload session")

    temp_path = _split_upload_temp_path(upload_dir, normalized_upload_id, suffix)
    if not temp_path.exists():
        raise HTTPException(status_code=404, detail="split upload file not found")

    written = 0
    with temp_path.open("ab") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
            written += len(chunk)

    meta["next_chunk_index"] = expected_index + 1
    _write_split_meta(upload_dir, normalized_upload_id, meta)

    return SplitUploadChunkResponse(
        upload_id=normalized_upload_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        received_bytes=written,
    )


@app.post("/upload/split/complete", response_model=UploadResponse)
async def upload_split_complete(payload: SplitUploadCompleteRequest) -> UploadResponse:
    upload_id = _safe_upload_id(payload.upload_id)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    meta = _read_split_meta(upload_dir, upload_id)
    total_chunks = meta.get("total_chunks")
    next_chunk_index = int(meta.get("next_chunk_index", 0))
    if total_chunks is None or next_chunk_index != int(total_chunks):
        raise HTTPException(status_code=400, detail="split upload is not complete")

    suffix = str(meta.get("suffix", "")).lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=500, detail="invalid split upload session")

    filename = Path(str(meta.get("filename", "")).strip()).name
    if not filename:
        raise HTTPException(status_code=500, detail="invalid split upload filename")

    temp_path = _split_upload_temp_path(upload_dir, upload_id, suffix)
    if not temp_path.exists():
        raise HTTPException(status_code=404, detail="split upload file not found")

    destination = upload_dir / f"{uuid4().hex}_{filename}"
    temp_path.replace(destination)

    meta_path = _split_upload_meta_path(upload_dir, upload_id)
    if meta_path.exists():
        meta_path.unlink()

    task = process_split_archive.delay(str(destination), 140)
    return UploadResponse(task_id=task.id)


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def status(task_id: str) -> TaskStatusResponse:
    result = AsyncResult(task_id, app=celery_app)
    state = result.state

    if state == "PENDING":
        return TaskStatusResponse(task_id=task_id, state=state, message="queued")

    if state == "PROGRESS":
        meta = result.info or {}
        return TaskStatusResponse(
            task_id=task_id,
            state=state,
            current=meta.get("current"),
            total=meta.get("total"),
            message=meta.get("message"),
        )

    if state == "SUCCESS":
        meta = result.result or {}
        return TaskStatusResponse(
            task_id=task_id,
            state=state,
            current=meta.get("current"),
            total=meta.get("total"),
            message=meta.get("message"),
            output_file=meta.get("output_file"),
        )

    if state == "FAILURE":
        return TaskStatusResponse(
            task_id=task_id,
            state=state,
            message=str(result.result),
        )

    meta = result.info or {}
    return TaskStatusResponse(
        task_id=task_id,
        state=state,
        current=meta.get("current"),
        total=meta.get("total"),
        message=meta.get("message"),
    )


@app.post("/tasks/{task_id}/cancel", response_model=TaskCancelResponse)
def cancel_task(task_id: str) -> TaskCancelResponse:
    """Send a revoke signal to the celery task."""
    celery_app.control.revoke(task_id, terminate=True)
    return TaskCancelResponse(task_id=task_id, message="task cancel signal sent")


@app.websocket("/ws/status/{task_id}")
async def websocket_status(websocket: WebSocket, task_id: str):
    await websocket.accept()
    # Create a new Redis connection for subscription
    r = redis.from_url(settings.celery_broker_url)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"task_progress:{task_id}")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                # Redis message data is bytes, decode it
                data = message["data"].decode("utf-8")
                await websocket.send_text(data)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe()
        await r.close()


@app.get("/export")
def export_data(task_id: str | None = None, db: Session = Depends(get_session)):
    import csv
    import io
    from fastapi.responses import StreamingResponse

    def iter_csv(data_records):
        output = io.StringIO()
        writer = csv.writer(output)
        # Write header
        writer.writerow(["id", "task_id", "row_number", "payload"])
        output.seek(0)
        yield output.read()
        output.truncate(0)
        output.seek(0)

        for record in data_records:
            writer.writerow(
                [record.id, record.task_id, record.row_number, record.payload]
            )
            output.seek(0)
            yield output.read()
            output.truncate(0)
            output.seek(0)

    query = db.query(DataRecord)
    if task_id:
        query = query.filter(DataRecord.task_id == task_id)

    records = query.order_by(DataRecord.id).all()

    filename = f"export_{task_id}.csv" if task_id else "export_all.csv"
    response = StreamingResponse(iter_csv(records), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@app.get("/download/{task_id}")
def download_result(task_id: str):
    output_dir = Path(settings.output_dir)
    result = AsyncResult(task_id, app=celery_app)
    if result.state == "SUCCESS":
        meta = result.result or {}
        output_file = meta.get("output_file")
        if output_file:
            file_path = output_dir / output_file
            if file_path.exists():
                return FileResponse(
                    path=file_path,
                    filename=file_path.name,
                    media_type=_detect_media_type(file_path),
                )

    candidates = list(output_dir.glob(f"{task_id}_desensitized.*"))
    if not candidates:
        split_zip = output_dir / f"{task_id}_split.zip"
        if split_zip.exists():
            return FileResponse(
                path=split_zip,
                filename=split_zip.name,
                media_type=_detect_media_type(split_zip),
            )
        raise HTTPException(status_code=404, detail="output file not ready")

    file_path = candidates[0]
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=_detect_media_type(file_path),
    )
