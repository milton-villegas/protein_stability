FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN VITE_API_URL="" npm run build

FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt smt

COPY core/ ./core/
COPY utils/ ./utils/
COPY config/ ./config/
COPY backend/ ./backend/

COPY --from=frontend-builder /app/frontend/build ./frontend/build

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
