/**
 * Minimal protocol-buffer wire-format writer.
 *
 * Only the encodings the ONNX exporter needs are implemented: varints,
 * length-delimited fields (strings, bytes, nested messages) and 32-bit floats.
 * There is no schema and no reflection — callers pass field numbers directly,
 * which is exactly how a hand-rolled serialiser for a fixed schema should work.
 */

export const WIRE_VARINT = 0;
export const WIRE_FIXED64 = 1;
export const WIRE_LENGTH = 2;
export const WIRE_FIXED32 = 5;

export class ProtoWriter {
  #buffer: Uint8Array;
  #length = 0;
  readonly #scratch = new DataView(new ArrayBuffer(8));

  constructor(capacity = 256) {
    this.#buffer = new Uint8Array(Math.max(16, capacity));
  }

  get length(): number {
    return this.#length;
  }

  #reserve(extra: number): void {
    const required = this.#length + extra;
    if (required <= this.#buffer.length) return;
    let capacity = this.#buffer.length * 2;
    while (capacity < required) capacity *= 2;
    const next = new Uint8Array(capacity);
    next.set(this.#buffer.subarray(0, this.#length));
    this.#buffer = next;
  }

  #byte(value: number): void {
    this.#reserve(1);
    this.#buffer[this.#length] = value & 0xff;
    this.#length += 1;
  }

  /** Base-128 varint. Negative values are encoded as their 64-bit two's complement. */
  writeVarint(value: number): void {
    if (value < 0 || !Number.isSafeInteger(value)) {
      let wide = BigInt.asUintN(64, BigInt(Math.trunc(value)));
      while (wide >= 0x80n) {
        this.#byte(Number(wide & 0x7fn) | 0x80);
        wide >>= 7n;
      }
      this.#byte(Number(wide));
      return;
    }
    let remaining = value;
    while (remaining >= 0x80) {
      this.#byte((remaining % 128) + 128);
      remaining = Math.floor(remaining / 128);
    }
    this.#byte(remaining);
  }

  writeTag(field: number, wireType: number): void {
    this.writeVarint(field * 8 + wireType);
  }

  /** int32 / int64 / bool / enum field. */
  varint(field: number, value: number): void {
    this.writeTag(field, WIRE_VARINT);
    this.writeVarint(value);
  }

  float(field: number, value: number): void {
    this.writeTag(field, WIRE_FIXED32);
    this.#scratch.setFloat32(0, value, true);
    this.#reserve(4);
    for (let i = 0; i < 4; i += 1) this.#byte(this.#scratch.getUint8(i));
  }

  bytes(field: number, data: Uint8Array): void {
    this.writeTag(field, WIRE_LENGTH);
    this.writeVarint(data.length);
    this.#reserve(data.length);
    this.#buffer.set(data, this.#length);
    this.#length += data.length;
  }

  string(field: number, value: string): void {
    this.bytes(field, new TextEncoder().encode(value));
  }

  /** Nested message, written as a length-delimited field. */
  message(field: number, sub: ProtoWriter): void {
    this.bytes(field, sub.toBytes());
  }

  /** A copy of everything written so far. */
  toBytes(): Uint8Array {
    return this.#buffer.slice(0, this.#length);
  }
}

/** Pack an array of float32 values into little-endian raw bytes. */
export function packFloat32(values: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i += 1) {
    view.setFloat32(i * 4, values[i], true);
  }
  return out;
}

/** Pack an array of int64 values into little-endian raw bytes. */
export function packInt64(values: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(values.length * 8);
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i += 1) {
    view.setBigInt64(i * 8, BigInt(Math.trunc(values[i])), true);
  }
  return out;
}
