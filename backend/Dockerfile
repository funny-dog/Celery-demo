FROM node:20-slim AS frontend-builder

WORKDIR /frontend

ARG NPM_REGISTRY=http://registry.npmmirror.com
ENV NODE_OPTIONS=--dns-result-order=ipv4first

COPY frontend/package*.json ./
COPY frontend/.npmrc ./
RUN npm config set registry ${NPM_REGISTRY}
RUN npm config set strict-ssl false
RUN npm config set fetch-retry-maxtimeout 60000
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

ENV UV_PROJECT_ENVIRONMENT=/app/.venv

COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen

COPY backend/ ./
COPY --from=frontend-builder /frontend/dist /app/frontend_dist

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
