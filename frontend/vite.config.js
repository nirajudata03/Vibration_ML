import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

function localServerStop() {
  return {
    name: 'local-server-stop',
    configureServer(server) {
      server.middlewares.use('/__local/stop', (request, response) => {
        if (request.method !== 'POST') {
          response.statusCode = 405;
          response.end('POST required');
          return;
        }
        const address = server.httpServer.address();
        const port = typeof address === 'object' && address ? address.port : null;
        const expectedOrigin = port ? `http://127.0.0.1:${port}` : '';
        if (request.headers.origin !== expectedOrigin) {
          response.statusCode = 403;
          response.end('Same-origin request required');
          return;
        }

        response.statusCode = 202;
        response.setHeader('Content-Type', 'application/json');
        response.end(JSON.stringify({ stopping: true }));
        setTimeout(() => {
          server.close().finally(() => process.exit(0));
        }, 700);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), localServerStop()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});
