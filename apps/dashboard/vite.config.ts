import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const api = process.env.DASHBOARD_API ?? "http://127.0.0.1:8002";

export default defineConfig({
	plugins: [react(), tailwindcss()],
	// Recharts is most of the bundle; the page is served locally, so one chunk is fine.
	build: { chunkSizeWarningLimit: 1200 },
	server: { proxy: { "/api": api } },
});
