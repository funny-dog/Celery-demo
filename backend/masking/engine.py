"""
Masking Rule Engine

Applies masking rules to data based on column/field configuration.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from .rules import MaskingStrategy, get_masking_strategy

# ---------------------------------------------------------------------------
# Safe condition evaluator – replaces the former eval()-based approach
# ---------------------------------------------------------------------------

# AST node types we allow in condition expressions.  Anything not listed here
# (e.g. ast.Call, ast.Attribute, ast.Lambda, ast.ListComp …) is rejected so
# that user-supplied strings cannot execute arbitrary code.
_ALLOWED_NODE_TYPES: frozenset[type] = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Constant,
        ast.Name,
        ast.Load,
        # Comparison operators
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    }
)

_CMP_OPERATORS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


class UnsafeExpressionError(Exception):
    """Raised when a condition expression contains disallowed constructs."""


def _validate_ast(node: ast.AST) -> None:
    """Walk *node* and raise if any disallowed AST node type is found."""
    for child in ast.walk(node):
        if type(child) not in _ALLOWED_NODE_TYPES:
            raise UnsafeExpressionError(
                f"Disallowed expression node: {type(child).__name__}"
            )


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    """Recursively evaluate a validated AST node against *variables*."""

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id == "None":
            return None
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        try:
            return variables[node.id]
        except KeyError:
            raise ValueError(f"Unknown column reference: {node.id!r}")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, variables)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_node(v, variables) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_node(v, variables) for v in node.values)

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, variables)
            op_func = _CMP_OPERATORS.get(type(op_node))
            if op_func is None:
                if isinstance(op_node, ast.In):
                    if right is None or left not in right:
                        return False
                elif isinstance(op_node, ast.NotIn):
                    if right is None or left in right:
                        return False
                else:
                    raise UnsafeExpressionError(
                        f"Unsupported comparison: {type(op_node).__name__}"
                    )
            else:
                if not op_func(left, right):
                    return False
            left = right
        return True

    raise UnsafeExpressionError(f"Cannot evaluate node: {type(node).__name__}")


def safe_eval_condition(condition: str, row: dict[str, Any]) -> bool:
    """Safely evaluate a condition expression against a row of data.

    Supports:
      - Comparison operators: ``==  !=  >  <  >=  <=``
      - Identity checks: ``is None``, ``is not None``
      - Logical operators: ``and``, ``or``, ``not``
      - String / numeric / None / bool literals
      - Column references (looked up in *row*)

    Raises ``UnsafeExpressionError`` if the expression contains disallowed
    constructs (function calls, attribute access, etc.).
    """
    tree = ast.parse(condition, mode="eval")
    _validate_ast(tree)
    return bool(_eval_node(tree, row))


class MaskingRule:
    """Represents a masking rule for a specific field/column."""

    def __init__(
        self,
        column_name: str,
        masking_type: str,
        params: dict[str, Any] | None = None,
        condition: str | None = None,
    ):
        """
        Initialize a masking rule.

        Args:
            column_name: Column/field name to apply masking to
            masking_type: Type of masking (email, phone, hash, etc.)
            params: Additional parameters for the masking strategy
            condition: Optional SQL-like condition for conditional masking
        """
        self.column_name = column_name
        self.masking_type = masking_type
        self.params = params or {}
        self.condition = condition
        self._strategy: MaskingStrategy | None = None

    @property
    def strategy(self) -> MaskingStrategy:
        """Lazy load the masking strategy."""
        if self._strategy is None:
            self._strategy = get_masking_strategy(self.masking_type)
        return self._strategy

    def apply(self, value: Any) -> Any:
        """Apply masking to a value."""
        return self.strategy.mask(value, self.params)

    def should_apply(self, row: dict[str, Any]) -> bool:
        """Check if masking should be applied based on condition."""
        if not self.condition:
            return True

        try:
            return safe_eval_condition(self.condition, row)
        except (UnsafeExpressionError, SyntaxError, ValueError):
            # If the condition is malformed or unsafe, default to applying
            # the rule (fail-closed).
            return True

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "column_name": self.column_name,
            "masking_type": self.masking_type,
            "params": self.params,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskingRule":
        """Create rule from dictionary."""
        return cls(
            column_name=data.get("column_name", ""),
            masking_type=data.get("masking_type", "redact"),
            params=data.get("params"),
            condition=data.get("condition"),
        )


class MaskingEngine:
    """Engine for applying masking rules to datasets."""

    def __init__(self):
        self.rules: dict[str, MaskingRule] = {}

    def add_rule(self, rule: MaskingRule) -> None:
        """Add a masking rule."""
        self.rules[rule.column_name.lower()] = rule

    def remove_rule(self, column_name: str) -> None:
        """Remove a masking rule."""
        self.rules.pop(column_name.lower(), None)

    def get_rule(self, column_name: str) -> MaskingRule | None:
        """Get a rule by column name."""
        return self.rules.get(column_name.lower())

    def apply_to_row(
        self, row: dict[str, Any], skip_unmatched: bool = False
    ) -> dict[str, Any]:
        """
        Apply masking rules to a row of data.

        Args:
            row: Dictionary of column -> value
            skip_unmatched: If True, only return columns with rules

        Returns:
            Masked row dictionary
        """
        result = {}

        for col_name, value in row.items():
            rule = self.get_rule(col_name)
            if rule:
                if rule.should_apply(row):
                    result[col_name] = rule.apply(value)
                else:
                    result[col_name] = value
            elif not skip_unmatched:
                result[col_name] = value

        return result

    def apply_to_rows(
        self, rows: list[dict[str, Any]], skip_unmatched: bool = False
    ) -> list[dict[str, Any]]:
        """Apply masking rules to multiple rows."""
        return [self.apply_to_row(row, skip_unmatched) for row in rows]

    def get_rules_summary(self) -> list[dict[str, Any]]:
        """Get summary of all rules."""
        return [rule.to_dict() for rule in self.rules.values()]

    def clear_rules(self) -> None:
        """Clear all rules."""
        self.rules.clear()

    @classmethod
    def from_rules_config(cls, rules_config: list[dict[str, Any]]) -> "MaskingEngine":
        """Create engine from rules configuration."""
        engine = cls()
        for rule_data in rules_config:
            rule = MaskingRule.from_dict(rule_data)
            engine.add_rule(rule)
        return engine
