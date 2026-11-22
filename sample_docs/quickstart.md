# TaskBoard — Local Quickstart Guide

This document outlines the developer onboarding and local setup process for the TaskBoard service.

---

## 1. Prerequisites

Before setting up TaskBoard, ensure your development machine meets the following requirements:
- **Python:** Version 3.11 or higher.
- **Package Manager:** `pip`.
- **SQLite:** Version 3.35 or higher (standard library support with JSON functions enabled).
- **Operating System:** Linux, macOS, or Windows Subsystem for Linux (WSL2).

---

## 2. Local Environment Setup

### 2.1 Clone Repository and Prepare Virtual Environment
Clone the repository and create an isolated virtual environment:

```bash
git clone https://github.com/example-org/taskboard.git
cd taskboard
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Configure Environment Variables
Copy the default environment template into a local `.env` file:

```bash
cp .env.example .env
```

Ensure the following default values are present in your `.env` file:
```env
TASKBOARD_ENV=development
TASKBOARD_HOST=0.0.0.0
TASKBOARD_PORT=8080
TASKBOARD_DATABASE_URL=sqlite:///./taskboard.db
TASKBOARD_DB_TIMEOUT=30
TASKBOARD_SECRET_KEY=dev-insecure-key-change-in-prod
```

---

## 3. Database Initialization

TaskBoard uses SQLite for storing task and project entities. To initialize the database schema and apply pending migrations:

```bash
python -m taskboard.cli db-init
```

This command creates the SQLite database file (`taskboard.db`) in the root directory and builds the required tables: `tasks`, `projects`, and `schema_migrations`.

---

## 4. Starting the API Server

To start the TaskBoard API server in local development mode with automatic reload:

```bash
taskboard api start --host 0.0.0.0 --port 8080
```

Alternatively, you can start the application directly using Uvicorn:

```bash
uvicorn taskboard.main:app --host 0.0.0.0 --port 8080 --reload
```

Once running, the API server listens on `http://0.0.0.0:8080`.

---

## 5. Verifying the Health Endpoint

Verify that the server is operational by calling the health probe:

```bash
curl -X GET http://localhost:8080/health
```

A healthy server responds with HTTP 200:
```json
{
  "status": "healthy",
  "version": "1.4.0",
  "database": "connected"
}
```
