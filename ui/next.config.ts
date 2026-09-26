import { existsSync } from "node:fs";
import { resolve } from "node:path";
import type { NextConfig } from "next";

// Local dev reads the repo-root .env (MONGODB_URI). On Vercel the file is absent
// and the variables come from the project settings.
const rootEnv = resolve(process.cwd(), "../.env");
if (!process.env.MONGODB_URI && existsSync(rootEnv)) process.loadEnvFile(rootEnv);

const nextConfig: NextConfig = {};

export default nextConfig;
