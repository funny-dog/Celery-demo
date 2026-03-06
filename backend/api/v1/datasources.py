from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import CurrentUser, get_current_user
from crypto import decrypt_sensitive_config, encrypt_sensitive_config
from database import get_session
from models import AuditLog, DataSource, DataSourceType

router = APIRouter(prefix="/datasources", tags=["Data Sources"])


# Pydantic schemas
class DataSourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    source_type: DataSourceType
    connection_config: dict[str, Any] = Field(default_factory=dict)


class DataSourceUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    connection_config: dict[str, Any] | None = None
    is_active: bool | None = None


class DataSourceResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None
    source_type: DataSourceType
    is_active: bool
    last_tested_at: datetime | None
    last_test_result: str | None
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    # Mask sensitive fields
    connection_config: dict[str, Any]

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj: DataSource) -> "DataSourceResponse":
        # Mask sensitive connection details (no need to decrypt – just mask them)
        config = obj.connection_config.copy() if obj.connection_config else {}
        sensitive_keys = [
            "password",
            "secret",
            "token",
            "api_key",
            "private_key",
            "secret_key",
            "access_key",
        ]
        for key in sensitive_keys:
            if key in config:
                config[key] = "***MASKED***"

        return cls(
            id=obj.id,
            tenant_id=obj.tenant_id,
            name=obj.name,
            description=obj.description,
            source_type=obj.source_type,
            is_active=obj.is_active,
            last_tested_at=obj.last_tested_at,
            last_test_result=obj.last_test_result,
            created_by=obj.created_by,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            connection_config=config,
        )


class DataSourceTestRequest(BaseModel):
    source_type: DataSourceType
    connection_config: dict[str, Any]


class DataSourceTestResponse(BaseModel):
    success: bool
    message: str
    details: dict[str, Any] | None = None


class DataSourceListResponse(BaseModel):
    total: int
    items: list[DataSourceResponse]


@router.get("", response_model=DataSourceListResponse)
def list_data_sources(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """List all data sources in the current tenant."""
    query = db.query(DataSource).filter(DataSource.tenant_id == current_user.tenant_id)

    if active_only:
        query = query.filter(DataSource.is_active == True)

    total = query.count()
    data_sources = (
        query.order_by(DataSource.created_at.desc()).offset(skip).limit(limit).all()
    )

    return DataSourceListResponse(
        total=total,
        items=[DataSourceResponse.from_orm(ds) for ds in data_sources],
    )


@router.get("/{source_id}", response_model=DataSourceResponse)
def get_data_source(
    source_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Get data source by ID."""
    data_source = (
        db.query(DataSource)
        .filter(
            DataSource.id == source_id,
            DataSource.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )

    return DataSourceResponse.from_orm(data_source)


@router.post("", response_model=DataSourceResponse, status_code=status.HTTP_201_CREATED)
def create_data_source(
    source_data: DataSourceCreate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Create a new data source."""
    # Check name uniqueness within tenant
    existing = (
        db.query(DataSource)
        .filter(
            DataSource.tenant_id == current_user.tenant_id,
            DataSource.name == source_data.name,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source with this name already exists",
        )

    # Validate connection config based on type
    _validate_connection_config(source_data.source_type, source_data.connection_config)

    # Encrypt sensitive fields before persisting
    encrypted_config = encrypt_sensitive_config(source_data.connection_config)

    data_source = DataSource(
        tenant_id=current_user.tenant_id,
        name=source_data.name,
        description=source_data.description,
        source_type=source_data.source_type,
        connection_config=encrypted_config,
        created_by=current_user.id,
    )
    db.add(data_source)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="create",
        resource_type="datasource",
        resource_id=str(data_source.id),
        details={
            "name": source_data.name,
            "source_type": source_data.source_type.value,
        },
    )
    db.add(audit)

    db.commit()
    db.refresh(data_source)

    return DataSourceResponse.from_orm(data_source)


@router.put("/{source_id}", response_model=DataSourceResponse)
def update_data_source(
    source_id: uuid.UUID,
    source_data: DataSourceUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Update data source."""
    data_source = (
        db.query(DataSource)
        .filter(
            DataSource.id == source_id,
            DataSource.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )

    update_dict = source_data.model_dump(exclude_unset=True)

    # Validate connection config if provided
    if "connection_config" in update_dict:
        _validate_connection_config(
            data_source.source_type, update_dict["connection_config"]
        )
        # Encrypt sensitive fields before persisting
        update_dict["connection_config"] = encrypt_sensitive_config(
            update_dict["connection_config"]
        )

    for field, value in update_dict.items():
        setattr(data_source, field, value)

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="update",
        resource_type="datasource",
        resource_id=str(data_source.id),
        details={"updated_fields": list(update_dict.keys())},
    )
    db.add(audit)

    db.commit()
    db.refresh(data_source)

    return DataSourceResponse.from_orm(data_source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_data_source(
    source_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Delete a data source."""
    data_source = (
        db.query(DataSource)
        .filter(
            DataSource.id == source_id,
            DataSource.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )

    # Audit log
    audit = AuditLog(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        action="delete",
        resource_type="datasource",
        resource_id=str(data_source.id),
        details={"name": data_source.name},
    )
    db.add(audit)

    db.delete(data_source)
    db.commit()

    return None


@router.post("/test", response_model=DataSourceTestResponse)
def test_data_source(
    test_request: DataSourceTestRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Test data source connection."""
    try:
        _validate_connection_config(
            test_request.source_type, test_request.connection_config
        )

        # Try to establish connection based on type
        if test_request.source_type == DataSourceType.MYSQL:
            _test_mysql_connection(test_request.connection_config)
        elif test_request.source_type == DataSourceType.POSTGRESQL:
            _test_postgresql_connection(test_request.connection_config)
        elif test_request.source_type == DataSourceType.MONGODB:
            _test_mongodb_connection(test_request.connection_config)
        elif test_request.source_type in (DataSourceType.S3, DataSourceType.MINIO):
            _test_s3_connection(
                test_request.source_type, test_request.connection_config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Connection test not implemented for {test_request.source_type}",
            )

        return DataSourceTestResponse(
            success=True,
            message="Connection successful",
            details={"tested_at": datetime.utcnow().isoformat()},
        )

    except Exception as e:
        return DataSourceTestResponse(
            success=False,
            message=str(e),
            details={"tested_at": datetime.utcnow().isoformat()},
        )


def _validate_connection_config(
    source_type: DataSourceType, config: dict[str, Any]
) -> None:
    """Validate required fields based on source type."""
    required_fields = {
        DataSourceType.MYSQL: ["host", "port", "database"],
        DataSourceType.POSTGRESQL: ["host", "port", "database"],
        DataSourceType.ORACLE: ["host", "port", "service_name"],
        DataSourceType.MONGODB: ["host", "port", "database"],
        DataSourceType.S3: ["bucket", "region"],
        DataSourceType.MINIO: ["bucket", "endpoint"],
        DataSourceType.FILE: [],
    }

    required = required_fields.get(source_type, [])
    missing = [f for f in required if f not in config or not config[f]]

    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(missing)}",
        )


def _test_mysql_connection(config: dict[str, Any]) -> None:
    """Test MySQL connection."""
    try:
        import pymysql
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pymysql not installed",
        )

    connection = pymysql.connect(
        host=config.get("host"),
        port=config.get("port", 3306),
        user=config.get("username", "root"),
        password=config.get("password", ""),
        database=config.get("database"),
        connect_timeout=5,
    )
    connection.close()


def _test_postgresql_connection(config: dict[str, Any]) -> None:
    """Test PostgreSQL connection."""
    connection = psycopg2.connect(
        host=config.get("host"),
        port=config.get("port", 5432),
        user=config.get("username", "postgres"),
        password=config.get("password", ""),
        database=config.get("database"),
        connect_timeout=5,
    )
    connection.close()


def _test_mongodb_connection(config: dict[str, Any]) -> None:
    """Test MongoDB connection."""
    try:
        from pymongo import MongoClient
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pymongo not installed",
        )

    client = MongoClient(
        host=config.get("host"),
        port=config.get("port", 27017),
        username=config.get("username"),
        password=config.get("password"),
        serverSelectionTimeoutMS=5000,
    )
    client.server_info()


def _test_s3_connection(source_type: DataSourceType, config: dict[str, Any]) -> None:
    """Test S3/MinIO connection."""
    try:
        import boto3
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="boto3 not installed",
        )

    client_kwargs = {
        "service_name": "s3",
        "region_name": config.get("region", "us-east-1"),
    }

    if source_type == DataSourceType.MINIO:
        client_kwargs["endpoint_url"] = config.get("endpoint")
        client_kwargs["aws_access_key_id"] = config.get("access_key")
        client_kwargs["aws_secret_access_key"] = config.get("secret_key")
        client_kwargs["verify"] = config.get("verify", True)
    else:
        client_kwargs["aws_access_key_id"] = config.get("access_key")
        client_kwargs["aws_secret_access_key"] = config.get("secret_key")

    client = boto3.client(**client_kwargs)
    client.head_bucket(Bucket=config.get("bucket"))
