"""The fixed train subset for gate checks (SPEC §4): six tasks, every slope and every friction at least once."""

from __future__ import annotations

from sim.tasks import task_id

GATE_TASKS: tuple[str, ...] = tuple(
    task_id("train", slope, friction)
    for slope, friction in (
        (0.0, 1.0),
        (2.0, 0.6),
        (4.0, 0.35),
        (6.0, 1.0),
        (2.0, 0.35),
        (6.0, 0.6),
    )
)
