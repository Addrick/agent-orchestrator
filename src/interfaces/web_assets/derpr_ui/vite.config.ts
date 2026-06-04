import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The portal is served by the FastAPI engine adapter under the `/derpr` path
// (GET /derpr + a StaticFiles mount at /derpr serving this build output).
// `base` must match so emitted asset URLs resolve under /derpr/.
// In dev (`npm run dev`) we keep base at "/" and proxy the API to the adapter
// on :5003 so the API client works against a live engine without CORS.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/derpr/' : '/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Disable CSS minify: Vite 8's default lightningcss mis-parses the ported
    // portal.css theme (dangling-combinator false positive) and esbuild isn't
    // bundled in this Vite build. Unminified CSS is fine for an internal tool.
    cssMinify: false,
  },
  server: {
    proxy: {
      '/api': { target: 'http://localhost:5003', changeOrigin: true },
      '/v1': { target: 'http://localhost:5003', changeOrigin: true },
    },
  },
}))
