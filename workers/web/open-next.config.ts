import { defineCloudflareConfig } from "@opennextjs/cloudflare";

// The landing page is prerendered; no ISR or external cache storage is needed.
export default defineCloudflareConfig();
