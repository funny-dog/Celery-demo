"""
Sensitive Data Discovery Scanner

Scans database tables and files to identify sensitive columns/fields
based on column names and data patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SensitivityLevel(str, Enum):
    """Sensitivity levels for discovered fields."""
    L1 = "L1"  # Low sensitivity
    L2 = "L2"  # Medium sensitivity
    L3 = "L3"  # High sensitivity
    L4 = "L4"  # Critical sensitivity


class DataType(str, Enum):
    """Detected data types."""
    EMAIL = "email"
    PHONE = "phone"
    ID_CARD = "id_card"
    BANK_CARD = "bank_card"
    SSN = "ssn"
    PASSPORT = "passport"
    NAME = "name"
    ADDRESS = "address"
    BIRTHDAY = "birthday"
    GENDER = "gender"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    UNKNOWN = "unknown"


# Sensitive field detection rules
SENSITIVE_RULES: dict[DataType, dict[str, Any]] = {
    DataType.EMAIL: {
        "keywords": ["email", "e-mail", "mail", "邮箱", "电邮"],
        "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        "sensitivity": SensitivityLevel.L3,
    },
    DataType.PHONE: {
        "keywords": [
            "phone", "mobile", "tel", "telephone", "手机号", "电话",
            "手机", "联系方式", "contact"
        ],
        "regex": r"^1[3-9]\d{9}$|^(\+?\d{1,4}[-.]?)?\(?\d{2,4}\)?[-.]?\d{3,4}[-.]?\d{4}$",
        "sensitivity": SensitivityLevel.L3,
    },
    DataType.ID_CARD: {
        "keywords": [
            "id_card", "idcard", "identity", "sfz", "身份证", "证件",
            "citizen_id", "national_id"
        ],
        "regex": r"^\d{17}[\dXx]$|^\d{8,10}$",
        "sensitivity": SensitivityLevel.L4,
    },
    DataType.BANK_CARD: {
        "keywords": [
            "bank_card", "bankcard", "debit_card", "银行卡", "储蓄卡",
            "account_number", "card_number"
        ],
        "regex": r"^\d{16,19}$",
        "sensitivity": SensitivityLevel.L4,
    },
    DataType.SSN: {
        "keywords": ["ssn", "social_security", "社保", "税号"],
        "regex": r"^\d{3}-\d{2}-\d{4}$|^\d{9,11}$",
        "sensitivity": SensitivityLevel.L4,
    },
    DataType.PASSPORT: {
        "keywords": ["passport", "travel_doc", "护照", "通行证"],
        "regex": r"^[A-Z0-9]{6,12}$",
        "sensitivity": SensitivityLevel.L4,
    },
    DataType.NAME: {
        "keywords": ["name", "full_name", "first_name", "last_name", "姓名", "用户名"],
        "sensitivity": SensitivityLevel.L3,
    },
    DataType.ADDRESS: {
        "keywords": ["address", "addr", "地址", "住址", "location"],
        "sensitivity": SensitivityLevel.L2,
    },
    DataType.BIRTHDAY: {
        "keywords": [
            "birthday", "birth", "dob", "生日", "出生",
            "age", "年龄"
        ],
        "regex": r"^\d{4}-\d{2}-\d{2}$|^\d{8}$",
        "sensitivity": SensitivityLevel.L2,
    },
    DataType.GENDER: {
        "keywords": ["gender", "sex", "性别", "男", "女"],
        "sensitivity": SensitivityLevel.L1,
    },
    DataType.IP_ADDRESS: {
        "keywords": ["ip", "ip_address", "IP 地址"],
        "regex": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        "sensitivity": SensitivityLevel.L2,
    },
    DataType.CREDIT_CARD: {
        "keywords": ["credit_card", "creditcard", "信用卡", "visa", "mastercard"],
        "regex": r"^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$",
        "sensitivity": SensitivityLevel.L4,
    },
}


@dataclass
class DiscoveredField:
    """Represents a discovered sensitive field."""
    table_name: str | None
    column_name: str
    data_type: DataType
    sensitivity: SensitivityLevel
    sample_value: str | None = None
    match_reason: str = ""  # keyword or regex match
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class ScanResult:
    """Result of a discovery scan."""
    total_columns: int = 0
    sensitive_columns: int = 0
    fields: list[DiscoveredField] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def add_field(self, field: DiscoveredField) -> None:
        self.fields.append(field)
        self.sensitive_columns += 1
        level = field.sensitivity.value
        self.summary[level] = self.summary.get(level, 0) + 1


def _normalize_column_name(name: str) -> str:
    """Normalize column name for matching."""
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def _match_by_keyword(column_name: str, keywords: list[str]) -> tuple[bool, float]:
    """
    Match column name against keywords.
    Returns (matched, confidence).
    """
    normalized = _normalize_column_name(column_name)

    for keyword in keywords:
        norm_keyword = keyword.lower().replace("_", "").replace("-", "")
        if norm_keyword in normalized:
            return True, 0.9

        # Partial match
        if len(keyword) >= 3 and keyword.lower() in normalized:
            return True, 0.7

    return False, 0.0


def _match_by_regex(value: str | None, pattern: str) -> tuple[bool, float]:
    """
    Match value against regex pattern.
    Returns (matched, confidence).
    """
    if not value:
        return False, 0.0

    try:
        if re.match(pattern, str(value).strip()):
            return True, 0.95
    except re.error:
        pass

    return False, 0.0


def discover_sensitive_field(
    column_name: str,
    sample_value: str | None = None,
) -> DiscoveredField | None:
    """
    Discover if a column is sensitive based on name and optional sample value.

    Args:
        column_name: The column/field name
        sample_value: Optional sample value for regex matching

    Returns:
        DiscoveredField if sensitive, None otherwise
    """
    best_match: DiscoveredField | None = None
    best_confidence = 0.0

    for data_type, rule in SENSITIVE_RULES.items():
        keywords = rule.get("keywords", [])
        regex = rule.get("regex")
        sensitivity = rule.get("sensitivity", SensitivityLevel.L2)

        # Try keyword match
        keyword_matched, keyword_conf = _match_by_keyword(column_name, keywords)

        # Try regex match if sample value provided
        regex_matched, regex_conf = False, 0.0
        if regex and sample_value:
            regex_matched, regex_conf = _match_by_regex(sample_value, regex)

        # Calculate final confidence
        if keyword_matched and regex_matched:
            confidence = max(keyword_conf, regex_conf) + 0.05  # Bonus for double match
            confidence = min(confidence, 1.0)
            match_reason = "keyword+regex"
        elif keyword_matched:
            confidence = keyword_conf
            match_reason = "keyword"
        elif regex_matched:
            confidence = regex_conf * 0.9  # Slightly lower for regex-only
            match_reason = "regex"
        else:
            continue

        if confidence > best_confidence:
            best_confidence = confidence
            best_match = DiscoveredField(
                table_name=None,
                column_name=column_name,
                data_type=data_type,
                sensitivity=sensitivity,
                sample_value=sample_value[:50] + "..." if sample_value and len(sample_value) > 50 else sample_value,
                match_reason=match_reason,
                confidence=confidence,
            )

    return best_match


def scan_table_schema(
    table_name: str,
    columns: list[dict[str, Any]],
    samples: dict[str, str | None] | None = None,
) -> ScanResult:
    """
    Scan a table schema for sensitive columns.

    Args:
        table_name: Table name
        columns: List of column info [{"name": "col1", "type": "varchar"}, ...]
        samples: Optional sample values {"col1": "sample_value", ...}

    Returns:
        ScanResult with discovered sensitive fields
    """
    result = ScanResult(total_columns=len(columns))

    for col in columns:
        col_name = col.get("name", "")
        sample = (samples or {}).get(col_name)

        discovered = discover_sensitive_field(col_name, sample)
        if discovered:
            discovered.table_name = table_name
            result.add_field(discovered)

    return result


def get_sensitivity_distribution(result: ScanResult) -> dict[str, float]:
    """Calculate sensitivity level distribution as percentages."""
    if not result.sensitive_columns:
        return {}

    total = result.sensitive_columns
    return {
        level: (count / total) * 100
        for level, count in result.summary.items()
    }
