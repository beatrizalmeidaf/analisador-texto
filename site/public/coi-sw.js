// Service worker de "cross-origin isolation".
//
// O GitHub Pages não permite configurar cabeçalhos HTTP. Sem os cabeçalhos
// COOP/COEP o navegador não libera SharedArrayBuffer, e o ONNX Runtime (que
// executa os modelos de IA) fica limitado a uma única thread. Este worker
// intercepta as respostas da própria página e acrescenta esses cabeçalhos,
// permitindo inferência multithread.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const req = event.request;
  // requisições "only-if-cached" de outra origem lançam erro se repassadas
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  // só respostas da própria origem precisam dos cabeçalhos; o resto segue direto
  if (new URL(req.url).origin !== self.location.origin) return;

  event.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      headers.set('Cross-Origin-Resource-Policy', 'same-origin');
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }),
  );
});
