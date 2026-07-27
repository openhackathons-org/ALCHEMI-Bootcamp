"""Support code for the Part 1 single-system scaling lesson.

The package keeps imports lazy so :mod:`aux.domain.config` remains usable by
the standard-library-only campaign planner.  Model composition and distributed
execution remain visible in the notebook.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import (
    DOMAIN_METHODOLOGY,
    DOMAIN_METHODOLOGY_SCHEMA,
    DomainMethodologyConfig,
)


_LAZY_EXPORTS = {
    "DomainLessonView": (".results", "DomainLessonView"),
    "MolecularBoxPlan": (".packing", "MolecularBoxPlan"),
    "PrebuiltDomainBoxBundle": (".prebuilt", "PrebuiltDomainBoxBundle"),
    "PrebuiltDomainBoxError": (".prebuilt", "PrebuiltDomainBoxError"),
    "box_summary_table": (".packing", "box_summary_table"),
    "build_nci_molecular_box": (".packing", "build_nci_molecular_box"),
    "load_prebuilt_domain_box": (".prebuilt", "load_prebuilt_domain_box"),
    "load_domain_lesson_view": (".results", "load_domain_lesson_view"),
    "plan_nci_molecular_box": (".packing", "plan_nci_molecular_box"),
}


def __getattr__(name: str) -> Any:
    """Load ASE/pandas-backed helpers only when a caller requests them."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = (
    "DOMAIN_METHODOLOGY",
    "DOMAIN_METHODOLOGY_SCHEMA",
    "DomainLessonView",
    "DomainMethodologyConfig",
    "MolecularBoxPlan",
    "PrebuiltDomainBoxBundle",
    "PrebuiltDomainBoxError",
    "box_summary_table",
    "build_nci_molecular_box",
    "load_prebuilt_domain_box",
    "load_domain_lesson_view",
    "plan_nci_molecular_box",
)
