import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { NextConfig } from 'next';

/**
 * The user's home directory contains an unrelated lockfile, which makes
 * Turbopack infer the wrong workspace root. Pin it to the monorepo root.
 */
const monorepoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

/**
 * GitHub Pages serves a project site from a subpath, so every asset URL needs a
 * prefix. Set NEXT_PUBLIC_BASE_PATH in CI (the workflow derives it from the
 * repository name); locally it stays empty so `next dev` works at the root.
 */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

const nextConfig: NextConfig = {
  output: 'export',
  basePath,
  assetPrefix: basePath || undefined,
  reactStrictMode: true,
  images: { unoptimized: true },
  // Workspace packages ship raw TypeScript rather than build output, so Next
  // compiles them as part of the app. This removes an entire build stage from
  // the monorepo and keeps types live across package boundaries.
  transpilePackages: [
    '@neuroforge/shared',
    '@neuroforge/math',
    '@neuroforge/shaders',
    '@neuroforge/simulation',
    '@neuroforge/renderer',
    '@neuroforge/physics',
    '@neuroforge/editor',
    '@neuroforge/io',
    '@neuroforge/ai',
    '@neuroforge/ui',
  ],
  typescript: { ignoreBuildErrors: false },
  turbopack: { root: monorepoRoot },
  experimental: {
    optimizePackageImports: ['lucide-react', '@react-three/drei'],
  },
};

export default nextConfig;
