/**
 * Incremental server-sent-event decoder.
 *
 * Both providers stream `text/event-stream`, and the FastAPI proxy relays those
 * bytes untouched, so one decoder serves all three transports. Frames are split
 * on a blank line; `data:` lines within a frame are joined with newlines, as the
 * SSE specification requires.
 */

export interface SseFrame {
  /** The `event:` field, or null when the frame did not carry one. */
  event: string | null;
  data: string;
}

export class SseDecoder {
  private buffer = '';

  /** Feed a decoded text chunk and take every frame it completed. */
  push(chunk: string): SseFrame[] {
    this.buffer += chunk;
    const frames: SseFrame[] = [];
    for (;;) {
      const match = /\r?\n\r?\n/.exec(this.buffer);
      if (match === null) break;
      const raw = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const frame = parseFrame(raw);
      if (frame !== null) frames.push(frame);
    }
    return frames;
  }

  /** Take whatever is left once the stream ends without a trailing blank line. */
  flush(): SseFrame[] {
    const rest = this.buffer;
    this.buffer = '';
    const frame = parseFrame(rest);
    return frame === null ? [] : [frame];
  }
}

function parseFrame(raw: string): SseFrame | null {
  if (raw.trim().length === 0) return null;
  let event: string | null = null;
  const data: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line.length === 0 || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const rest = colon === -1 ? '' : line.slice(colon + 1);
    const value = rest.startsWith(' ') ? rest.slice(1) : rest;
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
  }
  if (data.length === 0 && event === null) return null;
  return { event, data: data.join('\n') };
}
