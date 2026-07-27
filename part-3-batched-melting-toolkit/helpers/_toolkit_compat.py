"""Runtime compatibility helpers for the Toolkit execution path."""

from __future__ import annotations

from typing import Any


def _method_from_preallocated_neighbor_kwargs(kwargs: dict[str, Any]) -> str:
    """Mirror Toolkit's hook pre-allocation choice as an explicit Ops method."""

    has_batch = kwargs.get("batch_ptr") is not None or kwargs.get("batch_idx") is not None
    if kwargs.get("cells_per_dimension") is not None:
        return "batch_cell_list" if has_batch else "cell_list"
    if (
        kwargs.get("shift_range_per_dimension") is not None
        or kwargs.get("max_atoms_per_system") is not None
    ):
        return "batch_naive" if has_batch else "naive"
    return "batch_naive" if has_batch else "naive"


def apply_toolkit_runtime_compatibility() -> None:
    """Apply quiet runtime compatibility fixes for the pinned Toolkit stack."""

    try:
        import nvalchemi.hooks.neighbor_list as neighbor_hook_module
    except Exception:
        return

    current = getattr(neighbor_hook_module, "neighbor_list", None)
    if current is None or getattr(current, "_alchemi_explicit_method_patch", False):
        return

    def _neighbor_list_with_explicit_method(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("method") is None:
            kwargs["method"] = _method_from_preallocated_neighbor_kwargs(kwargs)
        return current(*args, **kwargs)

    _neighbor_list_with_explicit_method._alchemi_explicit_method_patch = True
    neighbor_hook_module.neighbor_list = _neighbor_list_with_explicit_method
