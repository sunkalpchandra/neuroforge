/**
 * Resolve hook that lets Node run the workspace's TypeScript sources directly.
 *
 * Two things the bundler does for free and Node does not: extensionless relative
 * imports ('./octree'), and the @neuroforge/* package aliases. Combined with
 * --experimental-strip-types this makes every package runnable under plain node,
 * which is what the verification scripts rely on.
 *
 * Usage: node --experimental-strip-types --import ./scripts/ts-register.mjs file.ts
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const packagesRoot = path.resolve(here, '..', 'packages');

const CANDIDATE_SUFFIXES = ['.ts', '.tsx', '/index.ts', '/index.tsx'];

function firstExisting(basePath) {
  for (const suffix of CANDIDATE_SUFFIXES) {
    const candidate = basePath + suffix;
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
  }
  return null;
}

export async function resolve(specifier, context, nextResolve) {
  // @neuroforge/<pkg> and @neuroforge/<pkg>/<subpath>
  if (specifier.startsWith('@neuroforge/')) {
    const rest = specifier.slice('@neuroforge/'.length);
    const slash = rest.indexOf('/');
    const pkg = slash === -1 ? rest : rest.slice(0, slash);
    const subpath = slash === -1 ? 'index' : rest.slice(slash + 1);
    const resolved = firstExisting(path.join(packagesRoot, pkg, 'src', subpath));
    if (resolved) return nextResolve(pathToFileURL(resolved).href, context);
  }

  // Extensionless relative imports.
  if (
    (specifier.startsWith('./') || specifier.startsWith('../')) &&
    !/\.[cm]?[jt]sx?$/.test(specifier) &&
    context.parentURL
  ) {
    const basePath = fileURLToPath(new URL(specifier, context.parentURL));
    const resolved = firstExisting(basePath);
    if (resolved) return nextResolve(pathToFileURL(resolved).href, context);
  }

  // Bare package specifiers from a script living outside the repo. Node resolves
  // node_modules by walking up from the importing file, which finds nothing when
  // the verification script sits in a scratch directory.
  if (!specifier.startsWith('.') && !specifier.startsWith('/') && !specifier.startsWith('node:')) {
    try {
      return await nextResolve(specifier, context);
    } catch (error) {
      // Retry with a parent inside the repo so Node's own resolver walks up to
      // the workspace node_modules and honours the package's exports map, which
      // a direct directory URL would bypass.
      const anchor = pathToFileURL(path.join(here, 'ts-loader.mjs')).href;
      if (context.parentURL !== anchor) {
        return nextResolve(specifier, { ...context, parentURL: anchor });
      }
      throw error;
    }
  }

  return nextResolve(specifier, context);
}
