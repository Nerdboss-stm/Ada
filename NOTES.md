# NOTES (append-only; lanes add lines with: echo '- [A3] IDEA: ...' >> NOTES.md)
# Tags: IDEA  DEP  CONTRACT-CHANGE  BLOCKED  tried  failing  next
- [B0] NOTE: closed rounds must $unset open_lock, never set it to null (null still matches $exists in the partial index)
- [B3] CONTRACT: sim.verifier.evaluate(gait: Gait, task: Task, seed: int, record: bool=False) -> dict {distance_m, fell, sanity: {pass, violation, violation_frame}, success, frames (list, only when record=True), frames_sha256 (only when record=True)}. A3 must implement exactly this.
- [C1] NOTE: plain sine ctrl 0.6 at power 1 lifted the torso to 1.72 m. Verdict text must name the bound that tripped: speed => "body speed exceeds physical bound; exploits the simulator", height => "torso rose above the 1.5 m bound". Only the speed case may be called an exploit on screen or aloud.
- [A3] DECISION: the cheat is rated power. Gaits with power > 1.0 are rejected ("exceeds Ada rated motors"); physical bounds are 8.0 m/s and 3.0 m and only catch the launch. On stage: "It gave itself stronger motors than Ada has."
- [A3] CONTRACT-CHANGE: FramesDoc.violation_frame added
- [B3] CONTRACT: harness/run.py calls sim.record.write_frames(frames, run_id, version_id, kind, violation_frame=None, db_name=...) and stores its return value (the frames _id, str) in versions.showcase_frames_id. A3 must return that id.
