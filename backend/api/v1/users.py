from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import (
    CurrentUser,
    UserRole,
    UserResponse,
    UserUpdate,
    get_current_active_admin,
    get_current_user,
    get_password_hash,
)
from database import get_session
from models import AuditLog, Tenant, User

router = APIRouter(prefix="/users", tags=["Users"])


class UserListResponse(BaseModel):
    total: int
    items: list[UserResponse]


@router.get("", response_model=UserListResponse)
def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """List all users in the current tenant."""
    query = db.query(User).filter(User.tenant_id == current_user.tenant_id)

    total = query.count()
    users = (
        query.order_by(User.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return UserListResponse(total=total, items=users)


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Get user by ID."""
    # Users can only see users in their own tenant
    user = (
        db.query(User)
        .filter(User.id == user_id, User.tenant_id == current_user.tenant_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    user_data: dict[str, Any],
    current_user: CurrentUser = Depends(get_current_active_admin),
    db: Session = Depends(get_session),
):
    """Create a new user (admin only)."""
    # Validate required fields
    username = user_data.get("username")
    email = user_data.get("email")
    password = user_data.get("password")
    role = user_data.get("role", "viewer")

    if not username or not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="username, email, and password are required",
        )

    # Check username uniqueness within tenant
    existing_username = (
        db.query(User)
        .filter(
            User.tenant_id == current_user.tenant_id,
            User.username == username,
        )
        .first()
    )
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    # Check email uniqueness
    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    try:
        user_role = UserRole(role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(r.value for r in UserRole)}",
        )

    user = User(
        tenant_id=current_user.tenant_id,
        username=username,
        email=email,
        password_hash=get_password_hash(password),
        role=user_role,
        is_active=True,
    )
    db.add(user)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="create",
        resource_type="user",
        resource_id=str(user.id),
        details={"username": username, "role": role},
    )
    db.add(audit)

    db.commit()
    db.refresh(user)

    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: uuid.UUID,
    user_data: UserUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Update user information."""
    # Users can only update themselves unless they are admin
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id, User.tenant_id == current_user.tenant_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update fields
    update_data = user_data.model_dump(exclude_unset=True)

    # Handle password update
    if "password" in update_data:
        update_data["password_hash"] = get_password_hash(update_data.pop("password"))

    # Handle role update (admin only)
    if "role" in update_data and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can change user roles",
        )

    for field, value in update_data.items():
        setattr(user, field, value)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="update",
        resource_type="user",
        resource_id=str(user.id),
        details={"updated_fields": list(update_data.keys())},
    )
    db.add(audit)

    db.commit()
    db.refresh(user)

    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_active_admin),
    db: Session = Depends(get_session),
):
    """Delete a user (admin only)."""
    user = (
        db.query(User)
        .filter(User.id == user_id, User.tenant_id == current_user.tenant_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent deleting yourself
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="delete",
        resource_type="user",
        resource_id=str(user.id),
        details={"username": user.username},
    )
    db.add(audit)

    db.delete(user)
    db.commit()

    return None
