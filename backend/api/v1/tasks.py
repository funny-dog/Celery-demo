from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import CurrentUser, get_current_user
from database import get_session
from models import AuditLog, DataSource, DataSourceType, DesensitizeTask, TaskStatus
from celery_app import celery_app
from celery.result import AsyncResult
from worker import process_db_desensitize, process_task_desensitize, process_desensitize

router = APIRouter(prefix="/tasks", tags=["Tasks"])


# Pydantic schemas
class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    source_id: uuid.UUID | None = None
    source_type: str = Field(..., description="db, file, s3")
    source_config: dict[str, Any] = Field(default_factory=dict)
    target_type: str = Field(..., description="db, file, s3")
    target_config: dict[str, Any] = Field(default_factory=dict)
    rules: dict[str, Any] = Field(default_factory=dict)


class TaskUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class TaskResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None
    source_id: uuid.UUID | None
    source_type: str
    target_type: str
    status: TaskStatus
    progress: float
    message: str | None
    error_detail: str | None
    input_rows: int | None
    output_rows: int | None
    masked_fields: int | None
    created_by: uuid.UUID | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    total: int
    items: list[TaskResponse]


class TaskStatusResponse(BaseModel):
    status: TaskStatus
    progress: float
    message: str | None
    error_detail: str | None
    current: int | None
    total: int | None


@router.get("", response_model=TaskListResponse)
def list_tasks(
    skip: int = 0,
    limit: int = 100,
    status_filter: TaskStatus | None = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """List all tasks in the current tenant."""
    query = db.query(DesensitizeTask).filter(
        DesensitizeTask.tenant_id == current_user.tenant_id
    )

    if status_filter:
        query = query.filter(DesensitizeTask.status == status_filter)

    total = query.count()
    tasks = (
        query.order_by(DesensitizeTask.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return TaskListResponse(total=total, items=tasks)


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Get task by ID."""
    task = (
        db.query(DesensitizeTask)
        .filter(
            DesensitizeTask.id == task_id,
            DesensitizeTask.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return task


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(
    task_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Get task execution status."""
    task = (
        db.query(DesensitizeTask)
        .filter(
            DesensitizeTask.id == task_id,
            DesensitizeTask.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    # If task has a celery task ID, get latest status from Celery
    if task.celery_task_id and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        try:
            result = AsyncResult(task.celery_task_id, app=celery_app)

            if result.state == "PENDING":
                pass  # Keep existing status
            elif result.state == "PROGRESS":
                meta = result.info or {}
                task.progress = meta.get("current", 0) / max(meta.get("total", 1), 1)
                task.message = meta.get("message")
            elif result.state == "SUCCESS":
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                meta = result.result or {}
                task.message = meta.get("message")
                task.input_rows = meta.get("input_rows")
                task.output_rows = meta.get("output_rows")
                task.masked_fields = meta.get("masked_fields")
                task.completed_at = datetime.utcnow()
            elif result.state == "FAILURE":
                task.status = TaskStatus.FAILED
                task.error_detail = str(result.result)

            db.commit()
        except Exception:
            pass  # Keep existing status on error

    return TaskStatusResponse(
        status=task.status,
        progress=task.progress,
        message=task.message,
        error_detail=task.error_detail,
        current=int(task.progress * 100),
        total=100,
    )


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(
    task_data: TaskCreate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Create a new desensitization task."""
    # Validate source if provided
    if task_data.source_id:
        source = (
            db.query(DataSource)
            .filter(
                DataSource.id == task_data.source_id,
                DataSource.tenant_id == current_user.tenant_id,
            )
            .first()
        )
        if not source:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data source ID",
            )

    # Create task in DB
    task = DesensitizeTask(
        tenant_id=current_user.tenant_id,
        name=task_data.name,
        description=task_data.description,
        source_id=task_data.source_id,
        source_type=task_data.source_type,
        source_config=task_data.source_config,
        target_type=task_data.target_type,
        target_config=task_data.target_config,
        rules=task_data.rules,
        status=TaskStatus.PENDING,
        progress=0.0,
        created_by=current_user.id,
    )
    db.add(task)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="create",
        resource_type="task",
        resource_id=str(task.id),
        details={
            "name": task_data.name,
            "source_type": task_data.source_type,
            "target_type": task_data.target_type,
        },
    )
    db.add(audit)

    db.commit()
    db.refresh(task)

    # Dispatch the appropriate Celery task based on source_type
    celery_result = None
    if task_data.source_type == "db":
        celery_result = process_db_desensitize.delay(str(task.id))
    elif task_data.source_type == "file":
        celery_result = process_task_desensitize.delay(str(task.id))

    if celery_result is not None:
        task.celery_task_id = celery_result.id
        task.status = TaskStatus.RUNNING
        db.commit()
        db.refresh(task)

    return task


@router.put("/{task_id}", response_model=TaskResponse)
def update_task(
    task_id: uuid.UUID,
    task_data: TaskUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Update task details (only if not running)."""
    task = (
        db.query(DesensitizeTask)
        .filter(
            DesensitizeTask.id == task_id,
            DesensitizeTask.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    # Can only update if not running
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update a running task",
        )

    update_dict = task_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(task, field, value)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="update",
        resource_type="task",
        resource_id=str(task.id),
        details={"updated_fields": list(update_dict.keys())},
    )
    db.add(audit)

    db.commit()
    db.refresh(task)

    return task


@router.post("/{task_id}/cancel", response_model=TaskResponse)
def cancel_task(
    task_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Cancel a running task."""
    task = (
        db.query(DesensitizeTask)
        .filter(
            DesensitizeTask.id == task_id,
            DesensitizeTask.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel task with status: {task.status}",
        )

    # Revoke Celery task if running
    if task.celery_task_id:
        celery_app.control.revoke(task.celery_task_id, terminate=True)

    task.status = TaskStatus.CANCELLED

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="cancel",
        resource_type="task",
        resource_id=str(task.id),
        details={},
    )
    db.add(audit)

    db.commit()
    db.refresh(task)

    return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Delete a task."""
    task = (
        db.query(DesensitizeTask)
        .filter(
            DesensitizeTask.id == task_id,
            DesensitizeTask.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    # Cannot delete running tasks
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running task",
        )

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="delete",
        resource_type="task",
        resource_id=str(task.id),
        details={"name": task.name},
    )
    db.add(audit)

    db.delete(task)
    db.commit()

    return None


@router.post("/{task_id}/retry", response_model=TaskResponse)
def retry_task(
    task_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Retry a failed task."""
    task = (
        db.query(DesensitizeTask)
        .filter(
            DesensitizeTask.id == task_id,
            DesensitizeTask.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    if task.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only retry failed or cancelled tasks",
        )

    # Reset task status
    task.status = TaskStatus.PENDING
    task.progress = 0.0
    task.message = None
    task.error_detail = None
    task.started_at = None
    task.completed_at = None
    task.celery_task_id = None

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="retry",
        resource_type="task",
        resource_id=str(task.id),
        details={},
    )
    db.add(audit)

    db.commit()
    db.refresh(task)

    # Re-dispatch the Celery task
    celery_result = None
    if task.source_type == "db":
        celery_result = process_db_desensitize.delay(str(task.id))
    elif task.source_type == "file":
        celery_result = process_task_desensitize.delay(str(task.id))

    if celery_result is not None:
        task.celery_task_id = celery_result.id
        task.status = TaskStatus.RUNNING
        db.commit()
        db.refresh(task)

    return task
