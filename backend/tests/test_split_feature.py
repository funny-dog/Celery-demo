import io
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
from worker import split_file_and_build_zip


def _set_temp_dirs(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "output_dir", str(output_dir))
    return upload_dir, output_dir


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("sample.txt", "text/plain"), ("sample.pdf", "application/pdf")],
)
def test_upload_split_accepts_txt_and_pdf(tmp_path, monkeypatch, filename, content_type):
    _set_temp_dirs(tmp_path, monkeypatch)
    client = TestClient(app)

    class DummyTask:
        id = "task-split-123"

    with patch("main.process_split_archive.delay", return_value=DummyTask()) as mock_delay:
        response = client.post(
            "/upload/split",
            files={"file": (filename, b"dummy-data", content_type)},
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-split-123"}

    saved_path = Path(mock_delay.call_args.args[0])
    assert saved_path.exists()
    assert saved_path.suffix.lower() in {".txt", ".pdf"}


def test_upload_split_rejects_unsupported_file_type(tmp_path, monkeypatch):
    _set_temp_dirs(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/upload/split",
        files={"file": ("sample.csv", b"a,b,c", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "only .pdf or .txt is supported"


def test_download_uses_task_output_file_from_success_meta(tmp_path, monkeypatch):
    _, output_dir = _set_temp_dirs(tmp_path, monkeypatch)
    client = TestClient(app)

    zip_path = output_dir / "task-split-zip.zip"
    zip_path.write_bytes(b"PK\x03\x04dummy")

    class FakeResult:
        state = "SUCCESS"
        result = {"output_file": zip_path.name}

    with patch("main.AsyncResult", return_value=FakeResult()):
        response = client.get("/download/task-id-1")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert zip_path.name in response.headers.get("content-disposition", "")


def test_download_falls_back_to_split_zip_when_result_backend_missing(tmp_path, monkeypatch):
    _, output_dir = _set_temp_dirs(tmp_path, monkeypatch)
    client = TestClient(app)
    task_id = "task-id-zip-fallback"

    zip_path = output_dir / f"{task_id}_split.zip"
    zip_path.write_bytes(b"PK\\x03\\x04dummy")

    class FakeResult:
        state = "PENDING"
        result = None

    with patch("main.AsyncResult", return_value=FakeResult()):
        response = client.get(f"/download/{task_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert zip_path.name in response.headers.get("content-disposition", "")


def test_split_file_and_build_zip_for_txt(tmp_path):
    source = tmp_path / "big.txt"
    source.write_bytes(b"A" * 25)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path, part_paths = split_file_and_build_zip(
        source,
        output_dir,
        task_id="task-text-1",
        chunk_size_bytes=10,
    )

    assert zip_path.exists()
    assert len(part_paths) == 3
    assert all(part.suffix == ".txt" for part in part_paths)

    with ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        assert len(infos) == 3
        assert all(info.filename.endswith(".txt") for info in infos)
        assert all(info.file_size <= 10 for info in infos)


def test_split_file_and_build_zip_for_pdf(tmp_path):
    from pypdf import PdfReader, PdfWriter

    source = tmp_path / "big.pdf"
    writer = PdfWriter()
    for _ in range(6):
        writer.add_blank_page(width=595, height=842)
    with source.open("wb") as handle:
        writer.write(handle)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path, part_paths = split_file_and_build_zip(
        source,
        output_dir,
        task_id="task-pdf-1",
        chunk_size_bytes=900,
    )

    assert zip_path.exists()
    assert len(part_paths) >= 2
    assert all(part.suffix == ".pdf" for part in part_paths)

    with ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        assert len(infos) == len(part_paths)
        assert all(info.filename.endswith(".pdf") for info in infos)
        for info in infos:
            data = archive.read(info.filename)
            reader = PdfReader(io.BytesIO(data))
            assert len(reader.pages) >= 1
