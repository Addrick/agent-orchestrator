import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The portal is served by the FastAPI engine adapter under the `/derpr` path
// (GET /derpr + a StaticFiles mount at /derpr serving this build output).
// `base` must match so emitted asset URLs resolve under /derpr/.
// In dev (`npm run dev`) we keep base at "/" and proxy the API to the adapter
// on :5003 so the API client works against a live engine without CORS.
// `DERPR_DEV_API` overrides the dev proxy target — useful for pointing the dev
// server at an engine on another host, or straight at a KoboldCPP backend when
// only the passthrough routes (e.g. /api/extra/perf) are under test.
const DEV_API = process.env.DERPR_DEV_API || 'http://localhost:5003'

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/derpr/' : '/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': { target: DEV_API, changeOrigin: true },
      '/v1': { target: DEV_API, changeOrigin: true },
      '/voice': { target: DEV_API, changeOrigin: true, ws: true },
    },
  },
}))
