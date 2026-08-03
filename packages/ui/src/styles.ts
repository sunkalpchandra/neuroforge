'use client';

import { useInsertionEffect } from 'react';

/**
 * Shared style fragments.
 *
 * `ease-out-expo` and the colour names resolve against the `@theme` block in the
 * host application's stylesheet; this package never ships its own Tailwind
 * config, so the tokens are the contract between the two.
 */

/** Ring drawn only for keyboard users, offset against the app background. */
export const FOCUS_RING =
  'outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-bg';

/** Same, for controls flush against a panel edge where an offset ring would clip. */
export const FOCUS_RING_INSET =
  'outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-inset';

/**
 * The only properties this library transitions. Layout-affecting properties are
 * deliberately absent: motion is expressed with `transform` and `opacity` so it
 * stays on the compositor.
 */
const TRANSITION_PROPERTIES =
  'transition-[transform,opacity,color,background-color,border-color,box-shadow]';

/** 120ms — micro-interactions: hover, focus, icon state. */
export const MOTION_FAST = `${TRANSITION_PROPERTIES} duration-[120ms] ease-out-expo`;

/** 160ms — the house default for control state changes. */
export const MOTION_BASE = `${TRANSITION_PROPERTIES} duration-[160ms] ease-out-expo`;

/** 200ms — larger surfaces and sliding indicators. */
export const MOTION_SLOW = `${TRANSITION_PROPERTIES} duration-[200ms] ease-out-expo`;

/** Applied to anything that must not be selected while being dragged. */
export const NO_SELECT = 'select-none [-webkit-user-drag:none]';

const STYLE_ELEMENT_ID = 'nf-ui-runtime-styles';

/**
 * Keyframes and the handful of rules Tailwind utilities cannot express (enter and
 * exit animations keyed off Radix `data-state`, and the alpha checkerboard).
 *
 * The sheet is injected once per document rather than shipped as a CSS file
 * because this package is consumed as source and has no build step of its own;
 * requiring the host app to import an extra stylesheet would be a second, easily
 * forgotten integration step.
 */
const RUNTIME_STYLES = `
@keyframes nf-fade-in{from{opacity:0}to{opacity:1}}
@keyframes nf-fade-out{from{opacity:1}to{opacity:0}}
@keyframes nf-dialog-in{from{opacity:0;transform:translate3d(0,10px,0) scale(.96)}to{opacity:1;transform:translate3d(0,0,0) scale(1)}}
@keyframes nf-dialog-out{from{opacity:1;transform:translate3d(0,0,0) scale(1)}to{opacity:0;transform:translate3d(0,6px,0) scale(.98)}}
@keyframes nf-pop-in{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
@keyframes nf-pop-out{from{opacity:1;transform:scale(1)}to{opacity:0;transform:scale(.97)}}
@keyframes nf-toast-in{from{opacity:0;transform:translate3d(0,16px,0) scale(.97)}to{opacity:1;transform:translate3d(0,0,0) scale(1)}}
.nf-anim-overlay[data-state='open']{animation:nf-fade-in 180ms cubic-bezier(.16,1,.3,1)}
.nf-anim-overlay[data-state='closed']{animation:nf-fade-out 140ms cubic-bezier(.65,0,.35,1)}
.nf-anim-dialog[data-state='open']{animation:nf-dialog-in 240ms cubic-bezier(.16,1,.3,1)}
.nf-anim-dialog[data-state='closed']{animation:nf-dialog-out 150ms cubic-bezier(.65,0,.35,1)}
.nf-anim-pop{transform-origin:var(--radix-popover-content-transform-origin,var(--radix-tooltip-content-transform-origin,var(--radix-select-content-transform-origin,var(--radix-dropdown-menu-content-transform-origin,center))))}
.nf-anim-pop[data-state='open'],.nf-anim-pop[data-state='delayed-open']{animation:nf-pop-in 140ms cubic-bezier(.16,1,.3,1)}
.nf-anim-pop[data-state='closed'],.nf-anim-pop[data-state='instant-open']{animation:nf-pop-out 110ms cubic-bezier(.65,0,.35,1)}
.nf-anim-toast-in{animation:nf-toast-in 260ms cubic-bezier(.16,1,.3,1)}
.nf-checker{background-image:linear-gradient(45deg,rgb(255 255 255/.12) 25%,transparent 25%),linear-gradient(-45deg,rgb(255 255 255/.12) 25%,transparent 25%),linear-gradient(45deg,transparent 75%,rgb(255 255 255/.12) 75%),linear-gradient(-45deg,transparent 75%,rgb(255 255 255/.12) 75%);background-size:8px 8px;background-position:0 0,0 4px,4px -4px,-4px 0}
@media (prefers-reduced-motion:reduce){.nf-anim-overlay,.nf-anim-dialog,.nf-anim-pop,.nf-anim-toast-in{animation-duration:1ms}}
`;

/**
 * Ensures the runtime stylesheet exists. Uses `useInsertionEffect` so the rules
 * land before React commits styles that depend on them, which avoids a flash of
 * un-animated content on the first mounted overlay.
 */
export function useRuntimeStyles(): void {
  useInsertionEffect(() => {
    if (typeof document === 'undefined') return;
    if (document.getElementById(STYLE_ELEMENT_ID) !== null) return;
    const element = document.createElement('style');
    element.id = STYLE_ELEMENT_ID;
    element.textContent = RUNTIME_STYLES;
    document.head.appendChild(element);
  }, []);
}
