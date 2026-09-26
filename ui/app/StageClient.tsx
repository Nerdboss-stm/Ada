"use client";

import dynamic from "next/dynamic";

// WebGL only exists in the browser.
const Stage = dynamic(() => import("./Stage"), { ssr: false });

export default function StageClient() {
  return <Stage />;
}
