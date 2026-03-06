from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import CurrentUser, get_current_user
from database import get_session
from models import AuditLog

router = APIRouter(prefix="/audit-logs", tags=["Audit Logs"])


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    resource_type: str
    resource_id: str | None
    details: dict
    ip_address: str | None
    user_agent: str | None
    timestamp: datetime

    model_config = {"from_attributes": True}


class AuditLogListResponse(BaseModel):
    total: int
    items: list[AuditLogResponse]


@router.get("", response_model=AuditLogListResponse)
def list_audit_logs(
    skip: int = 0,
    limit: int = 100,
    user_id: uuid.UUID | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """List audit logs for the current tenant."""
    query = db.query(AuditLog).filter(
        AuditLog.tenant_id == current_user.tenant_id
    )

    if user_id:
        query = query.filter(AuditLog.user_id == user_id)

    if action:
        query = query.filter(AuditLog.action == action)

    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    if start_date:
        query = query.filter(AuditLog.timestamp >= start_date)

    if end_date:
        query = query.filter(AuditLog.timestamp <= end_date)

    total = query.count()
    logs = (
        query.order_by(AuditLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return AuditLogListResponse(total=total, items=logs)


@router.get("/actions")
def get_available_actions(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Get list of available action types."""
    actions = (
        db.query(AuditLog.action)
        .filter(AuditLog.tenant_id == current_user.tenant_id)
        .distinct()
        .all()
    )
    return {"actions": [a[0] for a in actions]}


@router.get("/resource-types")
def get_available_resource_types(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Get list of available resource types."""
    types = (
        db.query(AuditLog.resource_type)
        .filter(AuditLog.tenant_id == current_user.tenant_id)
        .distinct()
        .all()
    )
    return {"resource_types": [t[0] for t in types]}
