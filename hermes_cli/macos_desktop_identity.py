"""Shared macOS Desktop Designated Requirement classification."""

from __future__ import annotations

import re
from typing import Literal


DESKTOP_BUNDLE_ID = "com.nousresearch.hermes"


def classify_designated_requirement(
    output: str,
) -> Literal["certificate", "identifier"] | None:
    """Classify one promotable Desktop DR, rejecting ambiguous expressions."""
    designated_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().lower().startswith("designated =>")
    ]
    if len(designated_lines) != 1 or "cdhash" in output.lower():
        return None

    requirement = designated_lines[0].split("=>", 1)[1].strip()
    identifier = f'identifier "{DESKTOP_BUNDLE_ID}"'
    if requirement == identifier:
        return "identifier"

    requirement_lower = requirement.lower()
    if (
        requirement.startswith(identifier + " and ")
        and not re.search(r"\bor\b", requirement, re.IGNORECASE)
        and (
            "anchor " in requirement_lower
            or "certificate " in requirement_lower
        )
    ):
        return "certificate"
    return None
