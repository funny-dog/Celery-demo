"""
Discovery API

Endpoints for scanning data sources and discovering sensitive fields.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import CurrentUser, get_current_user
from connectors.factory import create_connector
from crypto import decrypt_sensitive_config
from database import get_session
from discovery.scanner import (
    DataType,
    DiscoveredField,
    SensitivityLevel,
    ScanResult,
    discover_sensitive_field,
    get_sensitivity_distribution,
    scan_table_schema,
)
from models import AuditLog, DataSource

router = APIRouter(prefix="/discovery", tags=["Discovery"])


class ScanRequest(BaseModel):
    """Request to scan a data source."""

    source_id: uuid.UUID | None = None
    connector_type: str | None = None  # mysql, postgresql
    connection_config: dict[str, Any] | None = None
    tables: list[str] | None = None  # Specific tables to scan, None = all
    include_samples: bool = True
    sample_size: int = Field(default=10, ge=1, le=100)


class ScanColumn(BaseModel):
    """Column information in scan result."""

    name: str
    type: str
    nullable: bool


class DiscoveredFieldResponse(BaseModel):
    """Discovered sensitive field response."""

    table_name: str | None
    column_name: str
    data_type: str
    sensitivity: str
    sample_value: str | None
    match_reason: str
    confidence: float


class TableScanResult(BaseModel):
    """Scan result for a single table."""

    table_name: str
    total_columns: int
    sensitive_columns: int
    columns: list[ScanColumn]
    discovered_fields: list[DiscoveredFieldResponse]
    distribution: dict[str, float]


class ScanResponse(BaseModel):
    """Complete scan response."""

    total_tables: int
    total_columns: int
    total_sensitive: int
    tables: list[TableScanResult]
    overall_distribution: dict[str, float]


class QuickDiscoverRequest(BaseModel):
    """Quick discover based on column names only."""

    columns: list[str] = Field(..., description="List of column names to analyze")


class QuickDiscoverResponse(BaseModel):
    """Quick discover response."""

    results: list[DiscoveredFieldResponse]
    summary: dict[str, int]


@router.post("/scan", response_model=ScanResponse)
def scan_data_source(
    request: ScanRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Scan a data source for sensitive fields.

    Can use either an existing data source ID or provide connection config directly.
    """
    config = request.connection_config
    connector_type = request.connector_type

    # If source_id provided, get connection details from DB
    if request.source_id:
        data_source = (
            db.query(DataSource)
            .filter(
                DataSource.id == request.source_id,
                DataSource.tenant_id == current_user.tenant_id,
            )
            .first()
        )
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Data source not found",
            )

        config = decrypt_sensitive_config(data_source.connection_config)
        connector_type = data_source.source_type.value

    if not config or not connector_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either source_id or connector_type+connection_config required",
        )

    try:
        # Connect to data source
        connector = create_connector(connector_type, config)

        with connector:
            # Test connection first
            success, message = connector.test_connection()
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Connection test failed: {message}",
                )

            # Get tables to scan
            tables_to_scan = request.tables or connector.get_tables()

            if not tables_to_scan:
                return ScanResponse(
                    total_tables=0,
                    total_columns=0,
                    total_sensitive=0,
                    tables=[],
                    overall_distribution={},
                )

            # Scan each table
            table_results: list[TableScanResult] = []
            total_columns = 0
            total_sensitive = 0
            all_distribution: dict[str, int] = {}

            for table_name in tables_to_scan:
                # Get schema
                columns = connector.get_columns(table_name)

                # Get samples if requested
                samples: dict[str, str | None] = {}
                if request.include_samples:
                    sample_data = connector.get_sample_data(
                        table_name, request.sample_size
                    )
                    # Get first row values for each column
                    if sample_data:
                        first_row = sample_data[0]
                        for col in columns:
                            val = first_row.get(col["name"])
                            samples[col["name"]] = str(val) if val else None

                # Run scanner
                scan_result = scan_table_schema(table_name, columns, samples)

                # Build result
                discovered_fields = [
                    DiscoveredFieldResponse(
                        table_name=f.table_name,
                        column_name=f.column_name,
                        data_type=f.data_type.value,
                        sensitivity=f.sensitivity.value,
                        sample_value=f.sample_value,
                        match_reason=f.match_reason,
                        confidence=f.confidence,
                    )
                    for f in scan_result.fields
                ]

                distribution = get_sensitivity_distribution(scan_result)

                table_results.append(
                    TableScanResult(
                        table_name=table_name,
                        total_columns=scan_result.total_columns,
                        sensitive_columns=scan_result.sensitive_columns,
                        columns=[
                            ScanColumn(
                                name=c["name"],
                                type=c["type"],
                                nullable=c["nullable"],
                            )
                            for c in columns
                        ],
                        discovered_fields=discovered_fields,
                        distribution=distribution,
                    )
                )

                total_columns += scan_result.total_columns
                total_sensitive += scan_result.sensitive_columns
                for level, count in scan_result.summary.items():
                    all_distribution[level.value] = (
                        all_distribution.get(level.value, 0) + count
                    )

            # Calculate overall distribution
            overall_distribution = {}
            if total_sensitive > 0:
                overall_distribution = {
                    level: (count / total_sensitive) * 100
                    for level, count in all_distribution.items()
                }

            # Audit log
            audit = AuditLog(
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                action="scan",
                resource_type="discovery",
                details={
                    "tables_scanned": len(tables_to_scan),
                    "sensitive_found": total_sensitive,
                },
            )
            db.add(audit)
            db.commit()

            return ScanResponse(
                total_tables=len(tables_to_scan),
                total_columns=total_columns,
                total_sensitive=total_sensitive,
                tables=table_results,
                overall_distribution=overall_distribution,
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {str(e)}",
        )


@router.post("/quick-discover", response_model=QuickDiscoverResponse)
def quick_discover(request: QuickDiscoverRequest):
    """
    Quickly discover sensitive fields based on column names only.

    Useful for pre-scan estimation without connecting to database.
    """
    results: list[DiscoveredFieldResponse] = []
    summary: dict[str, int] = {}

    for column_name in request.columns:
        discovered = discover_sensitive_field(column_name)
        if discovered:
            results.append(
                DiscoveredFieldResponse(
                    table_name=discovered.table_name,
                    column_name=discovered.column_name,
                    data_type=discovered.data_type.value,
                    sensitivity=discovered.sensitivity.value,
                    sample_value=discovered.sample_value,
                    match_reason=discovered.match_reason,
                    confidence=discovered.confidence,
                )
            )
            level = discovered.sensitivity.value
            summary[level] = summary.get(level, 0) + 1

    return QuickDiscoverResponse(results=results, summary=summary)


@router.get("/sensitivity-levels")
def get_sensitivity_levels():
    """Get available sensitivity levels."""
    return {
        "levels": [
            {
                "value": "L4",
                "label": "Critical",
                "description": "ID cards, bank cards, passwords",
            },
            {"value": "L3", "label": "High", "description": "Phone, email, names"},
            {"value": "L2", "label": "Medium", "description": "Address, birthday, IP"},
            {"value": "L1", "label": "Low", "description": "Gender, education"},
        ]
    }


@router.get("/data-types")
def get_data_types():
    """Get available data types that can be discovered."""
    return {
        "types": [
            {"value": t.value, "name": t.name.replace("_", " ").title()}
            for t in DataType
        ]
    }
