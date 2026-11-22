# TaskBoard — Troubleshooting & Error Reference

This guide provides troubleshooting steps and solutions for common database, network, and configuration errors encountered when running TaskBoard.

---

## 1. Common Database Errors

### 1.1 Error: `sqlite3.OperationalError: database is locked`
- **Symptom:** API requests fail with HTTP 500 or 503, and the application log displays `sqlite3.OperationalError: database is locked`.
- **Root Cause:** Multiple concurrent processes or worker threads attempted to write to SQLite simultaneously, exceeding the busy handler timeout threshold.
- **Resolution:**
  1. Increase the SQLite connection timeout in your `.env` file:
     ```env
     TASKBOARD_DB_TIMEOUT=30
     ```
  2. Enable Write-Ahead Logging (WAL) mode on the SQLite database to permit concurrent readers while writing:
     ```bash
     sqlite3 taskboard.db "PRAGMA journal_mode=WAL;"
     ```
  3. Ensure TaskBoard runs as a single Uvicorn process during local development (`--workers 1`).

---

### 1.2 Error: `DatabaseMigrationError: schema mismatch detected`
- **Symptom:** Server startup fails immediately with `DatabaseMigrationError: table tasks has no column named priority`.
- **Root Cause:** The local SQLite database schema is out of date compared to the latest application models.
- **Resolution:**
  1. Run the database migration utility:
     ```bash
     taskboard db migrate
     ```
  2. For a clean local development reset (wipes local test data):
     ```bash
     rm taskboard.db
     python -m taskboard.cli db-init
     ```

---

## 2. Configuration & Startup Errors

### 2.1 Error: `ConfigValidationError: TASKBOARD_PORT must be an integer between 1024 and 65535`
- **Symptom:** Application crashes during startup with a configuration validation error.
- **Root Cause:** The `TASKBOARD_PORT` variable in `.env` is either unset, non-numeric, or set to a privileged port below 1024 (e.g., `80`).
- **Resolution:** Open `.env` and set `TASKBOARD_PORT` to a valid non-privileged port:
  ```env
  TASKBOARD_PORT=8080
  ```

---

### 2.2 Error: `ConnectionRefusedError: Address already in use (port 8080)`
- **Symptom:** Uvicorn fails to bind the socket on port 8080 with `Errno 98: Address already in use`.
- **Root Cause:** Another instance of TaskBoard or an unrelated local service is already listening on port 8080.
- **Resolution:**
  1. Identify the process holding port 8080:
     ```bash
     lsof -i :8080
     ```
  2. Terminate the existing process or change the port in `.env`:
     ```env
     TASKBOARD_PORT=8081
     ```
  3. Restart TaskBoard:
     ```bash
     taskboard api start --port 8081
     ```

---

## 3. Health Probe Diagnostics

If the health check endpoint `GET /health` returns HTTP 503:
1. Verify that `taskboard.db` exists in the expected directory and has read/write filesystem permissions for the current user.
2. Check whether another process holds an exclusive write lock on `taskboard.db`.
3. Inspect `taskboard.log` for database initialization traces.
