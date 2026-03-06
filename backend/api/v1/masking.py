"""
Masking Rules API

Endpoints for managing masking rules and applying them to data.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import CurrentUser, get_current_user
from database import get_session
from masking.engine import MaskingEngine, MaskingRule
from masking.rules import MASKING_REGISTRY, MaskingType

router = APIRouter(prefix="/masking", tags=["Masking"])


class MaskingRuleRequest(BaseModel):
    """Request to create/update a masking rule."""
    column_name: str = Field(..., description="Column/field name")
    masking_type: str = Field(..., description="Type of masking")
    params: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = Field(None, description="Optional condition")


class MaskingRuleResponse(BaseModel):
    """Masking rule response."""
    column_name: str
    masking_type: str
    params: dict[str, Any]
    condition: str | None


class MaskingRulesBulkRequest(BaseModel):
    """Request to apply multiple rules at once."""
    rules: list[MaskingRuleRequest]


class MaskingRulesListResponse(BaseModel):
    """List of masking rules."""
    total: int
    items: list[MaskingRuleResponse]


class MaskingTypesResponse(BaseModel):
    """Available masking types."""
    types: list[dict[str, str]]


class ApplyMaskingRequest(BaseModel):
    """Request to apply masking to sample data."""
    rules: list[MaskingRuleRequest]
    data: list[dict[str, Any]] = Field(
        ...,
        description="Sample data to mask",
        example=[[{"name": "张三", "email": "test@example.com"}]]
    )


class ApplyMaskingResponse(BaseModel):
    """Result of applying masking."""
    original_count: int
    masked_count: int
    data: list[dict[str, Any]]


@router.get("/types", response_model=MaskingTypesResponse)
def get_masking_types():
    """Get available masking types."""
    return MaskingTypesResponse(
        types=[
            {"value": t.value, "name": t.name.replace("_", " ").title()}
            for t in MaskingType
        ]
    )


@router.get("/strategies")
def get_masking_strategies():
    """Get all available masking strategies with descriptions."""
    strategies = {
        "email": {
            "name": "Email Masking",
            "description": "john@example.com -> j***@example.com",
            "params": {},
        },
        "phone": {
            "name": "Phone Masking",
            "description": "13812345678 -> 138****5678",
            "params": {},
        },
        "id_card": {
            "name": "ID Card Masking",
            "description": "110101199001011234 -> 11************34",
            "params": {},
        },
        "name": {
            "name": "Name Masking",
            "description": "张三 -> 张*",
            "params": {},
        },
        "address": {
            "name": "Address Masking",
            "description": "Keep first 6 characters",
            "params": {},
        },
        "hash": {
            "name": "Hash (SHA256)",
            "description": "One-way hash with optional salt",
            "params": {"salt": "optional_salt_string"},
        },
        "replace": {
            "name": "Fixed Replace",
            "description": "Replace with fixed value",
            "params": {"replace_value": "***REDACTED***"},
        },
        "redact": {
            "name": "Complete Redact",
            "description": "Replace all with asterisks",
            "params": {},
        },
        "generalize_age": {
            "name": "Age Generalization",
            "description": "Convert age to ranges (18-29, 30-44, etc.)",
            "params": {},
        },
        "nullify": {
            "name": "Nullify",
            "description": "Set value to NULL/empty",
            "params": {},
        },
    }
    return {"strategies": strategies}


@router.post("/validate")
def validate_masking_rule(rule: MaskingRuleRequest):
    """Validate a masking rule without saving."""
    if rule.masking_type not in MASKING_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown masking type: {rule.masking_type}",
        )

    try:
        strategy = MASKING_REGISTRY[rule.masking_type]()
        # Test with sample value
        test_value = "test@example.com" if rule.masking_type == "email" else "test"
        result = strategy.mask(test_value, rule.params or {})
        return {
            "valid": True,
            "example": {
                "input": test_value,
                "output": result if isinstance(result, str) else str(result),
            },
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


@router.post("/apply", response_model=ApplyMaskingResponse)
def apply_masking(request: ApplyMaskingRequest):
    """
    Apply masking rules to sample data.

    Useful for testing rules before saving them.
    """
    # Build engine from rules
    engine = MaskingEngine()
    for rule_data in request.rules:
        rule = MaskingRule(
            column_name=rule_data.column_name,
            masking_type=rule_data.masking_type,
            params=rule_data.params,
            condition=rule_data.condition,
        )
        engine.add_rule(rule)

    # Apply to data
    masked_data = []
    for row in request.data:
        masked_row = engine.apply_to_row(row)
        masked_data.append(masked_row)

    return ApplyMaskingResponse(
        original_count=len(request.data),
        masked_count=len(masked_data),
        data=masked_data,
    )


@router.get("/rules/recommended")
def get_recommended_rules(
    discovered_fields: list[str] | None = None,
):
    """
    Get recommended masking rules based on discovered field types.

    Pass discovered data types to get appropriate masking recommendations.
    """
    recommendations = {
        "email": {"masking_type": "email", "params": {}},
        "phone": {"masking_type": "phone", "params": {}},
        "id_card": {"masking_type": "id_card", "params": {}},
        "name": {"masking_type": "name", "params": {}},
        "address": {"masking_type": "address", "params": {}},
        "bank_card": {"masking_type": "redact", "params": {}},
        "credit_card": {"masking_type": "redact", "params": {}},
        "ssn": {"masking_type": "hash", "params": {"salt": "your-secret-salt"}},
        "passport": {"masking_type": "redact", "params": {}},
        "birthday": {"masking_type": "generalize_age", "params": {}},
        "gender": {"masking_type": "nullify", "params": {}},
        "ip_address": {"masking_type": "mask", "params": {}},
    }

    if discovered_fields:
        return {
            "recommendations": {
                field: recommendations.get(field.lower(), {"masking_type": "redact"})
                for field in discovered_fields
            }
        }

    return {"recommendations": recommendations}
