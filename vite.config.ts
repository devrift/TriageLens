import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig(() => {
  const isLib = process.env.BUILD_LIB === 'true';

  return {
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
