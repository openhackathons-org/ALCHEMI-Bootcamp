"""Pareto-front and hypervolume helpers for the SEI challenge."""

from __future__ import annotations


Point2D = tuple[float, float]


def dominates(a: Point2D, b: Point2D) -> bool:
    """Return whether 2D maximization point ``a`` Pareto-dominates point ``b``."""
    return a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1])


def pareto_flags(points: list[Point2D]) -> list[bool]:
    """Return booleans marking nondominated points for 2D maximization."""
    flags: list[bool] = []
    for i, point in enumerate(points):
        flags.append(
            not any(dominates(other, point) for j, other in enumerate(points) if i != j)
        )
    return flags


def hypervolume_2d(points: list[Point2D]) -> float:
    """Return dominated area for 2D maximization against reference point ``(0, 0)``."""
    if not points:
        return 0.0
    clipped = [
        (min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y))))
        for x, y in points
    ]
    front = [point for point, keep in zip(clipped, pareto_flags(clipped)) if keep]
    front.sort(key=lambda point: (point[0], -point[1]))

    area = 0.0
    previous_x = 0.0
    for x, y in front:
        if x > previous_x:
            area += (x - previous_x) * y
            previous_x = x
    return float(area)
