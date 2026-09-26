import { setServers } from "node:dns";
import { MongoClient, type Db } from "mongodb";

// Opt-in: some networks (phone hotspots) mangle SRV replies for Node's resolver.
// ADA_DNS_SERVERS="1.1.1.1,8.8.8.8" routes Node DNS through those servers.
if (process.env.ADA_DNS_SERVERS) setServers(process.env.ADA_DNS_SERVERS.split(","));

// One cached client per server process (survives dev hot reloads).
const g = globalThis as unknown as { _adaMongo?: Promise<MongoClient> };

function client(): Promise<MongoClient> {
  const uri = process.env.MONGODB_URI;
  if (!uri) throw new Error("MONGODB_URI is not set");
  g._adaMongo ??= new MongoClient(uri).connect().catch((err) => {
    g._adaMongo = undefined;
    throw err;
  });
  return g._adaMongo;
}

export async function db(name: string): Promise<Db> {
  return (await client()).db(name);
}
