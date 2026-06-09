import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/static/webui/",
  build: {
    outDir: "../src/eaa_core/gui/static/webui",
    emptyOutDir: true,
  },
});
