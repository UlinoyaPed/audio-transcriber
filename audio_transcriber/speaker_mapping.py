"""Validation helpers for safe speaker-name permutations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional


def speaker_permutation_error(
    mapping: object,
    current_names: Sequence[str],
) -> Optional[str]:
    """Return why ``mapping`` is not a complete permutation, or ``None``."""
    if not isinstance(mapping, Mapping):
        return "mapping must be a JSON object"
    expected = set(current_names)
    keys = set(mapping.keys())
    values = list(mapping.values())
    try:
        value_set = set(values)
    except TypeError:
        return "mapping values must be speaker-name strings"
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        return f"mapping keys must exactly match current names; missing={missing}, extra={extra}"
    if value_set != expected:
        missing = sorted(expected - value_set)
        extra = sorted(value_set - expected)
        return (
            "mapping values must exactly match current names; "
            f"missing={missing}, extra={extra}"
        )
    if len(values) != len(value_set):
        return "mapping values must be unique"
    return None
