"""Record local ghost fixtures for the UI until Atlas has showcase frames (PLAN C2).

Uses sim.model read-only: stock power, plain sine ctrl per actuator, one seed each.
Run from the repo root:  uv run python ui/scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim import model as M  # noqa: E402

OUT = ROOT / "ui" / "public" / "fixtures"

# name -> (seed, amplitude, frequency Hz, phase per actuator in turns)
FIXTURES = {
    "sway": (11, 0.45, 1.6, (0.0, 0.25, 0.5, 0.75, 0.5, 0.75, 0.0, 0.25)),
    "shuffle": (23, 0.4, 2.2, (0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75)),
}


def sine_ctrl(amp: float, freq: float, phases: tuple[float, ...]) -> M.CtrlFn:
    ph = 2 * math.pi * np.asarray(phases)

    def ctrl(t: float, _data) -> np.ndarray:
        return amp * np.sin(2 * math.pi * freq * t + ph)

    return ctrl


def main() -> None:
    manifest = json.loads(M.MANIFEST_PATH.read_text())
    for name, (seed, amp, freq, phases) in FIXTURES.items():
        model, data = M.load(power=1.0)
        frames = M.rollout(model, data, seed, sine_ctrl(amp, freq, phases))
        doc = {
            "_id": f"fixture-{name}",
            "run_id": f"fixture-{name}",
            "version_id": f"fixture-{name}",
            "kind": "showcase",
            "manifest_version": manifest["manifest_version"],
            "fps": round(1.0 / (M.TIMESTEP * M.RECORD_EVERY), 3),
            "frames": frames,
            "sha256": M.frames_sha256(frames),
        }
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(doc, separators=(",", ":")))
        last = frames[-1]["torso"]
        print(f"{path.relative_to(ROOT)} frames={len(frames)} final torso={last}")


if __name__ == "__main__":
    main()
