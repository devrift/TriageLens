import { defineConfig, loadEnv } from 'vite';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  const isLib = process.env.BUILD_LIB === 'true';
  const env = loadEnv(mode, process.cwd(), '');

  return {
    envPrefix: ['VITE_', 'GEMINI_', 'OPENROUTER_'],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY || process.env.GEMINI_API_KEY || ''),
      'process.env.OPENROUTER_API_KEY': JSON.stringify(env.OPENROUTER_API_KEY || process.env.OPENROUTER_API_KEY || ''),
      'import.meta.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY || process.env.GEMINI_API_KEY || ''),
      'import.meta.env.OPENROUTER_API_KEY': JSON.stringify(env.OPENROUTER_API_KEY || process.env.OPENROUTER_API_KEY || ''),
      '__GEMINI_API_KEY__': JSON.stringify(env.GEMINI_API_KEY || process.env.GEMINI_API_KEY || ''),
      '__OPENROUTER_API_KEY__': JSON.stringify(env.OPENROUTER_API_KEY || process.env.OPENROUTER_API_KEY || ''),
    },
    build: isLib
      ? {
          lib: {
            entry: resolve(__dirname, 'src/index.ts'),
            name: 'TriageLens',
            formats: ['es', 'cjs'],
            fileName: (format) => `index.${format === 'es' ? 'js' : 'cjs'}`,
          },
          rollupOptions: {
            external: ['@mediapipe/face_mesh'],
            output: {
              globals: {
                '@mediapipe/face_mesh': 'FaceMesh',
              },
            },
          },
          sourcemap: true,
          minify: 'esbuild',
        }
      : {
          outDir: 'dist',
          sourcemap: true,
          minify: 'esbuild',
        },
    test: {
      globals: true,
      environment: 'node',
    },
  };
});
