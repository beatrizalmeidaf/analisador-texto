import { defineConfig } from 'vite';

export default defineConfig({
  // caminhos relativos: funciona em https://<usuario>.github.io/analisador-texto/ e em qualquer subpasta
  base: './',
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
  },
  // o onnxruntime-web carrega seus .wasm em tempo de execução; não deve ser pré-empacotado
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
});
