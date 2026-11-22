# TaskBoard — REST API Reference

The TaskBoard REST API provides endpoints for creating, listing, inspecting, and updating tasks in the TaskBoard system.

- **Base URL:** `http://localhost:8080/api/v1`
- **Default Content-Type:** `application/json`

---

## 1. System Health

### `GET /health`
Returns the operational status of the service and underlying database connection.

- **Request:** None
- **Responses:**
  - `200 OK`
    ```json
    {
      "status": "healthy",
      "version": "1.4.0",
      "database": "connected"
    }
    ```
  - `503 Service Unavailable`
    ```json
    {
      "status": "unhealthy",
      "version": "1.4.0",
      "database": "disconnected",
      "error": "database is locked"
    }
    ```

---

## 2. Task Management

### `POST /api/v1/tasks`
Create a new task on the board.

- **Headers:** `Content-Type: application/json`
- **Request Body Fields:**
  - `title` (string, required): Task title. Must be between 1 and 100 characters.
  - `description` (string, optional): Detailed task notes or markdown description.
  - `priority` (string, optional): Priority level. Allowed values: `"low"`, `"medium"`, `"high"`. Default: `"medium"`.
  - `assignee_id` (string, optional): Unique identifier of the assigned team member (e.g., `"usr_9823"`).
- **Request Example:**
  ```json
  {
    "title": "Configure SQLite connection pooling",
    "description": "Increase timeout to 30 seconds and enable WAL journal mode.",
    "priority": "high",
    "assignee_id": "usr_9823"
  }
  ```
- **Responses:**
  - `201 Created`
    ```json
    {
      "id": "tsk_01h8x9p4k3",
      "title": "Configure SQLite connection pooling",
      "description": "Increase timeout to 30 seconds and enable WAL journal mode.",
      "priority": "high",
      "status": "todo",
      "assignee_id": "usr_9823",
      "created_at": "2026-11-22T13:15:00Z"
    }
    ```
  - `400 Bad Request`
    ```json
    {
      "detail": "Field 'title' cannot be empty or exceed 100 characters"
    }
    ```

---

### `GET /api/v1/tasks`
Retrieve a paginated list of tasks with optional status filtering.

- **Query Parameters:**
  - `status` (string, optional): Filter by task status (`"todo"`, `"in_progress"`, `"done"`).
  - `limit` (integer, optional): Maximum items to return (default `20`, max `100`).
- **Responses:**
  - `200 OK`
    ```json
    {
      "tasks": [
        {
          "id": "tsk_01h8x9p4k3",
          "title": "Configure SQLite connection pooling",
          "priority": "high",
          "status": "todo",
          "assignee_id": "usr_9823",
          "created_at": "2026-11-22T13:15:00Z"
        }
      ],
      "total": 1
    }
    ```

---

### `PATCH /api/v1/tasks/{task_id}`
Update the status or fields of an existing task.

- **Path Parameters:**
  - `task_id` (string, required): The unique task ID.
- **Request Body Fields:**
  - `status` (string, optional): Updated status (`"todo"`, `"in_progress"`, `"done"`).
  - `priority` (string, optional): Updated priority (`"low"`, `"medium"`, `"high"`).
- **Responses:**
  - `200 OK`
    ```json
    {
      "id": "tsk_01h8x9p4k3",
      "status": "in_progress",
      "updated_at": "2026-11-22T13:20:00Z"
    }
    ```
  - `404 Not Found`
    ```json
    {
      "detail": "Task 'tsk_01h8x9p4k3' not found"
    }
    ```
