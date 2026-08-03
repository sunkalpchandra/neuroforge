import type { SimulationBuffers } from '@neuroforge/shared';

/**
 * Compressed sparse row adjacency over the synapse list.
 *
 * When a neuron spikes, the integrator needs its outgoing synapses; when a
 * neuron receives a spike, the plasticity rules need its incoming ones. Scanning
 * the whole synapse array for either would make a single spike O(synapses),
 * which is the difference between a network that runs and one that does not.
 *
 * Rebuilt only when topology changes — the engine calls invalidate() after any
 * structural edit, and the counts are checked as a cheap safety net in case it
 * forgets.
 */
export class Adjacency {
  private outStart = new Uint32Array(1);
  private outIndex = new Uint32Array(0);
  private inStart = new Uint32Array(1);
  private inIndex = new Uint32Array(0);

  private builtNeurons = -1;
  private builtSynapses = -1;
  private dirty = true;

  invalidate(): void {
    this.dirty = true;
  }

  ensure(buffers: SimulationBuffers): void {
    const neurons = buffers.neurons.count;
    const synapses = buffers.synapses.count;
    if (!this.dirty && neurons === this.builtNeurons && synapses === this.builtSynapses) return;
    this.build(buffers);
    this.builtNeurons = neurons;
    this.builtSynapses = synapses;
    this.dirty = false;
  }

  private build(buffers: SimulationBuffers): void {
    const neuronCount = buffers.neurons.count;
    const synapseCount = buffers.synapses.count;
    const { pre, post } = buffers.synapses;

    if (this.outStart.length < neuronCount + 1) {
      this.outStart = new Uint32Array(neuronCount + 1);
      this.inStart = new Uint32Array(neuronCount + 1);
    } else {
      this.outStart.fill(0, 0, neuronCount + 1);
      this.inStart.fill(0, 0, neuronCount + 1);
    }
    if (this.outIndex.length < synapseCount) {
      this.outIndex = new Uint32Array(synapseCount);
      this.inIndex = new Uint32Array(synapseCount);
    }

    // Counting pass, written one slot to the right so the prefix sum below
    // leaves each bucket's start index in place without a second shift.
    for (let s = 0; s < synapseCount; s += 1) {
      const a = pre[s];
      const b = post[s];
      if (a < neuronCount) this.outStart[a + 1] += 1;
      if (b < neuronCount) this.inStart[b + 1] += 1;
    }
    for (let i = 0; i < neuronCount; i += 1) {
      this.outStart[i + 1] += this.outStart[i];
      this.inStart[i + 1] += this.inStart[i];
    }

    const outCursor = new Uint32Array(neuronCount);
    const inCursor = new Uint32Array(neuronCount);
    for (let s = 0; s < synapseCount; s += 1) {
      const a = pre[s];
      const b = post[s];
      if (a < neuronCount) {
        this.outIndex[this.outStart[a] + outCursor[a]] = s;
        outCursor[a] += 1;
      }
      if (b < neuronCount) {
        this.inIndex[this.inStart[b] + inCursor[b]] = s;
        inCursor[b] += 1;
      }
    }
  }

  outBegin(neuron: number): number {
    return this.outStart[neuron];
  }

  outEnd(neuron: number): number {
    return this.outStart[neuron + 1];
  }

  outAt(index: number): number {
    return this.outIndex[index];
  }

  inBegin(neuron: number): number {
    return this.inStart[neuron];
  }

  inEnd(neuron: number): number {
    return this.inStart[neuron + 1];
  }

  inAt(index: number): number {
    return this.inIndex[index];
  }

  outDegree(neuron: number): number {
    return this.outStart[neuron + 1] - this.outStart[neuron];
  }

  inDegree(neuron: number): number {
    return this.inStart[neuron + 1] - this.inStart[neuron];
  }
}
