// Inserts one test edit (origin "cli") into ada.edits and walks its four gate stages
// (CONTRACTS.md §4) 1.5 s apart: each stage emits start, then pass 1.5 s later. Finally the edit
// is marked accepted with its actual delta. Prints the edit _id.
//   node scripts/emit_edit.mjs            # from ui/
//   node scripts/emit_edit.mjs --cleanup  # deletes exactly the documents this script wrote
import { setServers } from "node:dns";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { MongoClient } from "mongodb";

const SOURCE = "ui/scripts/emit_edit.mjs";
const ID_PREFIX = "cli-emit-";
const STAGES = ["gate.verifier", "gate.gpa", "gate.meta", "gate.constraints"];
const STEP_MS = 1500;

const envFile = resolve(import.meta.dirname, "../../.env");
if (!process.env.MONGODB_URI && existsSync(envFile)) process.loadEnvFile(envFile);
if (!process.env.MONGODB_URI) {
  console.error("MONGODB_URI is not set");
  process.exit(1);
}
if (process.env.ADA_DNS_SERVERS) setServers(process.env.ADA_DNS_SERVERS.split(","));

const client = new MongoClient(process.env.MONGODB_URI);
const ada = client.db("ada");

/** Only edits this script inserted: its _id prefix and origin "cli"; only their events it tagged. */
async function cleanup() {
  const ids = (
    await ada
      .collection("edits")
      .find({ _id: { $regex: `^${ID_PREFIX}\\d+$` }, origin: "cli" }, { projection: { _id: 1 } })
      .toArray()
  ).map((d) => d._id);
  if (ids.length === 0) return console.log("nothing to clean up");
  const events = await ada.collection("events").deleteMany({ edit_id: { $in: ids }, "payload.source": SOURCE });
  const edits = await ada.collection("edits").deleteMany({ _id: { $in: ids }, origin: "cli" });
  console.log(`deleted ${edits.deletedCount} edit(s), ${events.deletedCount} event(s): ${ids.join(", ")}`);
}

async function emit() {
  const latest = await ada.collection("versions").findOne({}, { sort: { created_at: -1 }, projection: { _id: 1 } });
  const editId = `${ID_PREFIX}${Date.now()}`;
  const fromVersion = latest ? String(latest._id) : "v0";
  await ada.collection("edits").insertOne({
    _id: editId,
    round_id: null,
    from_version: fromVersion,
    to_version: null,
    origin: "cli",
    primitive: "rules",
    old: "stride 0.20 m on every slope",
    new: "stride 0.14 m when slope_deg > 8",
    rationale: `Test edit written by ${SOURCE}, not by a model.`,
    predicted_delta: 0.05,
    actual_delta: null,
    verdict: null,
    reason: null,
    violation_frame: null,
    frames_id: null,
    created_at: new Date(),
  });
  console.log(editId);

  const event = (stage, status) =>
    ada.collection("events").insertOne({
      ts: new Date(),
      round_id: null,
      version_id: fromVersion,
      edit_id: editId,
      stage,
      status,
      payload: { source: SOURCE },
    });

  for (const stage of STAGES) {
    await sleep(STEP_MS);
    await event(stage, "start");
    console.log(`${stage} start`);
    await sleep(STEP_MS);
    await event(stage, "pass");
    console.log(`${stage} pass`);
  }
  await ada.collection("edits").updateOne({ _id: editId }, { $set: { verdict: "accepted", actual_delta: 0.04 } });
  console.log("accepted");
}

try {
  if (process.argv.includes("--cleanup")) await cleanup();
  else await emit();
} finally {
  await client.close();
}
