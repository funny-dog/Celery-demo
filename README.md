# Bulk Desensitizer

A full-stack demo showcasing a high-performance async architecture for CSV/XLSX desensitization.

## Tech Stack
- Backend: Python 3.11, FastAPI, Celery, Redis, SQLAlchemy (PostgreSQL)
- Frontend: Vue 3 (Vite)
- Infrastructure: Docker & Docker Compose

## Quick Start
```bash
docker compose up --build
```

Then open:
- Frontend: http://localhost:18000
- API docs: http://localhost:18000/docs

If `18000` is occupied, you can override it:
```bash
BACKEND_PORT=19000 docker compose up --build
```

## How It Works
1. Upload a CSV or XLSX via `POST /upload/desensitize`
2. FastAPI saves the file and triggers a Celery task (non-blocking)
3. Celery desensitizes sensitive columns and writes a new output file
4. Frontend polls `GET /status/{task_id}` every 2 seconds and updates the progress bar
5. Download the result via `GET /download/{task_id}`

For large file splitting:
1. Upload a PDF or TXT via `POST /upload/split`
2. Celery splits the file into chunks of up to 140MB each
3. Backend packs all split parts into a ZIP archive
4. Download the ZIP via `GET /download/{task_id}`

## API
- `POST /upload/desensitize` (multipart form-data, field: `file`)
- `POST /upload/split` (multipart form-data, field: `file`, supports `.pdf`/`.txt`)
- `GET /status/{task_id}`
- `GET /download/{task_id}`

## Desensitization Rules
The worker applies masking based on header keywords (case-insensitive):
- Email: `email`, `e-mail`, `mail`, `邮箱`
- Phone: `phone`, `mobile`, `tel`, `telephone`, `手机号`, `电话`
- ID: `id_card`, `idcard`, `identity`, `ssn`, `passport`, `身份证`, `证件`
- Name: `name`, `full_name`, `first_name`, `last_name`, `姓名`
- Address: `address`, `addr`, `地址`

If a column does not match any keyword, it is left unchanged.

## Frontend Usage
1. Open the frontend at http://localhost:5173
2. Upload a CSV/XLSX file with sensitive columns
3. Wait for the progress bar to reach 100%
4. Click "Download" to save the desensitized file

## Sample Data
A sample Excel file is provided at `sample_data.xlsx`.
The backend supports both CSV and XLSX, so the Excel file can be uploaded directly.

## Services (docker-compose)
- `db`: PostgreSQL 15
- `redis`: Redis 7
- `backend`: FastAPI app + embedded frontend static files
- `worker`: Celery worker

## Local Development (optional)
Backend:
```bash
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Worker:
```bash
cd backend
uv run celery -A worker.celery_app worker --loglevel=info
```

Frontend (optional local dev):
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
