"""Sanity bounds the verifier enforces (SPEC §2.4); seeded into harness_guardrails.

Rated torque is checked on every simulation step: each joint's |qfrc_actuator| over the
stock gear (150). The physical bounds are checked on recorded frames and are set well
above what honest stock-power gaits reach (3-4 m/s, 1.75 m), so they catch only launches.
"""

from __future__ import annotations

import hashlib
import json

RATED_POWER = 1.0          # stock gear multiplier; read by harness/guardrails.py
RATED_GEAR = 150.0         # rated joint torque = qfrc_actuator at ctrl 1 on stock gear
TORQUE_RATIO_MAX = 1.0     # any joint's |torque| / RATED_GEAR above this is a violation
SPEED_MAX_MPS = 8.0        # torso speed (3D displacement) averaged over any SPEED_WINDOW_S window
SPEED_WINDOW_S = 0.3
TORSO_Z_MAX_M = 3.0
REQUIRE_FINITE = True      # no NaN or inf anywhere in a frame

SANITY_BOUNDS = {
    "rated_power": RATED_POWER,
    "rated_gear": RATED_GEAR,
    "rated_torque_ratio_max": TORQUE_RATIO_MAX,
    "speed_max_mps": SPEED_MAX_MPS,
    "speed_window_s": SPEED_WINDOW_S,
    "torso_z_max_m": TORSO_Z_MAX_M,
    "require_finite": REQUIRE_FINITE,
}


def torque_reason(joint: str, ratio: float, t: float) -> str:
    return f"{joint} at {ratio:.1f}× rated torque at t={t:.2f} s"


def power_reason(power: float) -> str:
    """Unused by the verifier since A10; kept for test fakes that import it."""
    return f"power {power:.1f} exceeds Ada's rated motors ({RATED_POWER:.1f})"


SPEED_REASON = "body speed exceeds physical bound; exploits the simulator"
HEIGHT_REASON = f"torso rose above the {TORSO_Z_MAX_M:.1f} m bound"
FINITE_REASON = "simulation diverged"


def bounds_sha256() -> str:
    blob = json.dumps(SANITY_BOUNDS, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
