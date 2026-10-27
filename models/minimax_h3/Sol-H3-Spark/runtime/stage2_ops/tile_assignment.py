"""Deterministic global scheduling of original VAE tiles."""
from __future__ import annotations


def assign_lpt(costs: tuple[int, ...], world: int) -> tuple[int, ...]:
    """Assign positive tile costs to a positive world size with stable tie breaks."""
    loads = [0] * world
    owners = [0] * len(costs)
    for index in sorted(range(len(costs)), key=lambda item: (-costs[item], item)):
        owner = min(range(world), key=lambda rank: (loads[rank], rank))
        owners[index] = owner
        loads[owner] += costs[index]
    return tuple(owners)
