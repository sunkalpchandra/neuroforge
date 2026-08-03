'use client';

import { useSyncExternalStore } from 'react';

import {
  getCameraServerSnapshot,
  getCameraSnapshot,
  subscribeCamera,
} from '@/lib/runtime';

/** Target on-screen length of the scale bar, in CSS pixels. */
const TARGET_PX = 96;

/** 1-2-5 sequence, so the bar always reads as a round number. */
const NICE_STEPS = [1, 2, 5];

/**
 * Round a raw world-unit length down to the nearest 1, 2 or 5 times a power of
 * ten. A scale bar reading "1.73 units" is useless; the point of the widget is
 * that its label is a number you can multiply in your head.
 */
function niceLength(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const exponent = Math.floor(Math.log10(raw));
  const decade = 10 ** exponent;
  const normalised = raw / decade;
  let chosen = NICE_STEPS[0];
  for (const step of NICE_STEPS) {
    if (normalised >= step) chosen = step;
  }
  return chosen * decade;
}

function formatUnits(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}k u`;
  if (value >= 1) return `${value} u`;
  return `${value.toFixed(2)} u`;
}

/**
 * Scale bar and orientation gizmo.
 *
 * Both answer questions the 3D view cannot: how big is what I am looking at, and
 * which way am I facing. A connectome viewer without them leaves a user unable to
 * say whether two cells are close or the camera is simply zoomed in.
 */
export function ViewportHud() {
  const camera = useSyncExternalStore(
    subscribeCamera,
    getCameraSnapshot,
    getCameraServerSnapshot,
  );

  // World units spanned by one CSS pixel at the pivot distance, from the
  // perspective frustum: the visible height at distance d is 2*d*tan(fov/2).
  const visibleHeight = 2 * camera.distance * Math.tan(camera.fov / 2);
  const unitsPerPixel = visibleHeight / Math.max(1, camera.viewportHeight);
  const barUnits = niceLength(TARGET_PX * unitsPerPixel);
  const barPixels = Math.round(barUnits / unitsPerPixel);

  // The gizmo projects each world axis onto the view basis, so an axis pointing
  // at the camera collapses to a dot exactly as it should.
  const project = (axis: [number, number, number]): { x: number; y: number; z: number } => ({
    x: axis[0] * camera.right[0] + axis[1] * camera.right[1] + axis[2] * camera.right[2],
    y: axis[0] * camera.up[0] + axis[1] * camera.up[1] + axis[2] * camera.up[2],
    z:
      axis[0] * camera.forward[0] +
      axis[1] * camera.forward[1] +
      axis[2] * camera.forward[2],
  });

  const axes: { label: string; color: string; v: { x: number; y: number; z: number } }[] = [
    { label: 'X', color: '#FF5C7A', v: project([1, 0, 0]) },
    { label: 'Y', color: '#4ADE80', v: project([0, 1, 0]) },
    { label: 'Z', color: '#4FD1FF', v: project([0, 0, 1]) },
  ];
  // Draw the axis pointing away from the viewer first so nearer ones overlap it.
  const ordered = [...axes].sort((a, b) => a.v.z - b.v.z);
  const R = 15;

  return (
    <div className="pointer-events-none absolute bottom-3 left-1/2 flex -translate-x-1/2 items-end gap-3">
      <div className="nf-glass flex items-center gap-2 rounded-panel px-2 py-1.5">
        <div className="flex flex-col items-center gap-1">
          <span className="nf-numeric text-[10px] leading-none text-ink-muted">
            {formatUnits(barUnits)}
          </span>
          <svg width={barPixels} height={6} aria-hidden className="block">
            <line x1="0.5" y1="0" x2="0.5" y2="6" stroke="currentColor" className="text-ink-muted" />
            <line
              x1="0"
              y1="3"
              x2={barPixels}
              y2="3"
              stroke="currentColor"
              className="text-ink-muted"
            />
            <line
              x1={barPixels - 0.5}
              y1="0"
              x2={barPixels - 0.5}
              y2="6"
              stroke="currentColor"
              className="text-ink-muted"
            />
          </svg>
        </div>

        <div className="h-6 w-px bg-hairline" />

        <svg
          width={38}
          height={38}
          viewBox="-19 -19 38 38"
          aria-label="Camera orientation"
          role="img"
        >
          {ordered.map((axis) => {
            const x = axis.v.x * R;
            // SVG y grows downward; world up must point up on screen.
            const y = -axis.v.y * R;
            const facing = axis.v.z > 0 ? 0.35 : 1;
            return (
              <g key={axis.label} opacity={facing}>
                <line x1="0" y1="0" x2={x} y2={y} stroke={axis.color} strokeWidth="1.5" />
                <circle cx={x} cy={y} r="4.5" fill={axis.color} />
                <text
                  x={x}
                  y={y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize="6"
                  fill="#07090B"
                  fontWeight="600"
                >
                  {axis.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
