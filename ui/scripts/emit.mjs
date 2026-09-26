// Inserts one test event (CONTRACTS.md §4) into ada.events and prints its _id.
// Usage (from ui/): npm run emit
import { setServers } from "node:dns";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { MongoClient } from "mongodb";

const envFile = resolve(import.meta.dirname, "../../.env");
if (!process.env.MONGODB_URI && existsSync(envFile)) process.loadEnvFile(envFile);
if (!process.env.MONGODB_URI) {
  console.error("MONGODB_URI is not set");
  process.exit(1);
}

if (process.env.ADA_DNS_SERVERS) setServers(process.env.ADA_DNS_SERVERS.split(","));

const client = new MongoClient(process.env.MONGODB_URI);
try {
  const event = {
    ts: new Date().toISOString(),
    round_id: null,
    version_id: null,
    edit_id: null,
    stage: "harness",
    status: "info",
    payload: { source: "ui/scripts/emit.mjs", note: "C0 test event" },
  };
  const { insertedId } = await client.db("ada").collection("events").insertOne(event);
  console.log(insertedId.toHexString());
} finally {
  await client.close();
}
