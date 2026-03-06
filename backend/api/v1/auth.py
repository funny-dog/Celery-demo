from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from auth import (
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    CurrentUser,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    oauth2_scheme,
)
from config import settings
from database import get_session
from models import AuditLog, User
from token_blacklist import blacklist_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_session)):
    """Authenticate user and return access token."""
    user = authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Update last login
    user.last_login_at = datetime.utcnow()
    db.commit()

    # Create tokens
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "tenant_id": str(user.tenant_id),
            "role": user.role.value,
        }
    )
    refresh_token = create_refresh_token(
        data={
            "sub": str(user.id),
            "tenant_id": str(user.tenant_id),
        }
    )

    # Log login action
    audit = AuditLog(
        tenant_id=user.tenant_id,
        user_id=user.id,
        action="login",
        resource_type="user",
        resource_id=str(user.id),
        details={"method": "password"},
    )
    db.add(audit)
    db.commit()

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/logout")
def logout(
    token: str = Depends(oauth2_scheme),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Logout current user by blacklisting the access token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        if exp is not None:
            remaining = int(exp) - int(datetime.utcnow().timestamp())
            if remaining > 0:
                blacklist_token(token, remaining)
    except JWTError:
        # Token already validated by get_current_user; log and continue.
        logger.warning("Failed to decode token during logout", exc_info=True)

    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=Token)
def refresh_token(refresh_token: str, db: Session = Depends(get_session)):
    """Refresh access token using refresh token."""
    from jose import jwt, JWTError

    from config import settings

    try:
        payload = jwt.decode(
            refresh_token, settings.jwt_secret_key, algorithms=[ALGORITHM]
        )
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        user_id: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")

        if not user_id or not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or disabled",
            )

        # Create new tokens
        new_access_token = create_access_token(
            data={
                "sub": str(user.id),
                "tenant_id": str(user.tenant_id),
                "role": user.role.value,
            }
        )
        new_refresh_token = create_refresh_token(
            data={
                "sub": str(user.id),
                "tenant_id": str(user.tenant_id),
            }
        )

        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
def register(user_data: UserCreate, db: Session = Depends(get_session)):
    """Register a new user (for self-registration or admin creation)."""
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # For now, use default tenant - in production this would be handled differently
    from models import Tenant

    tenant = db.query(Tenant).first()
    if not tenant:
        # Create default tenant
        from auth import create_default_tenant, create_admin_user

        tenant = create_default_tenant(db)
        # Create admin user for the tenant
        admin_user = create_admin_user(db, tenant.id)
        # Don't allow creating another user with admin role if admin exists
        if user_data.role.value == "admin":
            user_data.role = user_data.role.__class__.OPERATOR

    # Check username uniqueness within tenant
    existing_username = (
        db.query(User)
        .filter(User.tenant_id == tenant.id, User.username == user_data.username)
        .first()
    )
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists in this organization",
        )

    # Create user
    user = User(
        tenant_id=tenant.id,
        username=user_data.username,
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        role=user_data.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: CurrentUser = Depends(get_current_user)):
    """Get current user information."""
    return current_user
