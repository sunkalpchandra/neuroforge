//! Topology analysis over the synapse columns, backing the inspector's network
//! analysis readouts.
//!
//! Every neuron slot below `count` becomes a node whose `NodeIndex` equals its
//! slot, so results index straight back into the SoA columns. Only enabled
//! synapses contribute edges: a disabled synapse is structurally absent, which is
//! what the editor means by toggling it off. Neuron `enabled` is deliberately
//! ignored — a dimmed neuron is still part of the topology being inspected.

use petgraph::Direction;
use petgraph::algo::{is_cyclic_directed, tarjan_scc};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;

pub type Topology = DiGraph<(), ()>;

/// Build the analysis graph from the raw columns. Out-of-range endpoints are
/// skipped rather than trusted: `pre` and `post` are written by JavaScript
/// directly into linear memory and carry no guarantee.
pub fn build(
    neuron_count: usize,
    synapse_count: usize,
    pre: &[u32],
    post: &[u32],
    enabled: &[u8],
) -> Topology {
    let mut g: Topology = DiGraph::with_capacity(neuron_count, synapse_count);
    for _ in 0..neuron_count {
        g.add_node(());
    }
    for i in 0..synapse_count {
        if enabled.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        let (Some(&a), Some(&b)) = (pre.get(i), post.get(i)) else {
            continue;
        };
        let (a, b) = (a as usize, b as usize);
        if a >= neuron_count || b >= neuron_count {
            continue;
        }
        g.add_edge(NodeIndex::new(a), NodeIndex::new(b), ());
    }
    g
}

/// Weakly connected components. Returns a dense component label per neuron, in
/// slot order, relabelled so that ids are 0..component_count.
pub fn components(g: &Topology) -> Vec<u32> {
    let n = g.node_count();
    let mut uf = UnionFind::<usize>::new(n);
    for e in g.edge_references() {
        uf.union(e.source().index(), e.target().index());
    }
    dense_labels((0..n).map(|i| uf.find(i)), n)
}

/// Strongly connected components. Every non-trivial component is a recurrent
/// motif: a set of neurons that can all drive each other.
pub fn strongly_connected_components(g: &Topology) -> Vec<u32> {
    let n = g.node_count();
    let mut labels = vec![0u32; n];
    for (id, group) in tarjan_scc(g).into_iter().enumerate() {
        for node in group {
            if let Some(slot) = labels.get_mut(node.index()) {
                *slot = id as u32;
            }
        }
    }
    labels
}

/// Neurons that sit inside a cycle: either a component of two or more neurons or
/// a self-connected one. This is the readout the inspector labels "recurrent".
pub fn recurrent_nodes(g: &Topology) -> Vec<u32> {
    let mut out = Vec::new();
    for group in tarjan_scc(g) {
        let recurrent = group.len() > 1
            || group
                .first()
                .is_some_and(|&n| g.find_edge(n, n).is_some());
        if recurrent {
            for node in group {
                out.push(node.index() as u32);
            }
        }
    }
    out.sort_unstable();
    out
}

pub fn distinct_count(labels: &[u32]) -> u32 {
    let mut seen: Vec<u32> = labels.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len() as u32
}

pub fn in_degrees(g: &Topology) -> Vec<u32> {
    (0..g.node_count())
        .map(|i| g.neighbors_directed(NodeIndex::new(i), Direction::Incoming).count() as u32)
        .collect()
}

pub fn out_degrees(g: &Topology) -> Vec<u32> {
    (0..g.node_count())
        .map(|i| g.neighbors_directed(NodeIndex::new(i), Direction::Outgoing).count() as u32)
        .collect()
}

/// Breadth-first shortest path along directed edges, returned as the full node
/// sequence from `from` to `to` inclusive. Empty when either endpoint is out of
/// range or no path exists; a single element when `from == to`.
pub fn shortest_path(g: &Topology, from: usize, to: usize) -> Vec<u32> {
    let n = g.node_count();
    if from >= n || to >= n {
        return Vec::new();
    }
    if from == to {
        return vec![from as u32];
    }
    const NONE: u32 = u32::MAX;
    let mut prev = vec![NONE; n];
    let mut visited = vec![false; n];
    let mut queue = std::collections::VecDeque::with_capacity(n);
    visited[from] = true;
    queue.push_back(from);
    while let Some(node) = queue.pop_front() {
        for next in g.neighbors_directed(NodeIndex::new(node), Direction::Outgoing) {
            let idx = next.index();
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            prev[idx] = node as u32;
            if idx == to {
                let mut path = vec![to as u32];
                let mut cursor = to;
                while cursor != from {
                    let p = prev[cursor];
                    if p == NONE {
                        return Vec::new();
                    }
                    cursor = p as usize;
                    path.push(cursor as u32);
                }
                path.reverse();
                return path;
            }
            queue.push_back(idx);
        }
    }
    Vec::new()
}

pub fn has_cycle(g: &Topology) -> bool {
    is_cyclic_directed(g)
}

/// Local clustering coefficient per neuron, computed on the undirected
/// projection: the fraction of a neuron's neighbour pairs that are themselves
/// connected in either direction. Nodes with fewer than two neighbours score 0.
pub fn clustering(g: &Topology) -> Vec<f32> {
    let n = g.node_count();
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
    for e in g.edge_references() {
        let (a, b) = (e.source().index(), e.target().index());
        if a == b {
            continue;
        }
        adjacency[a].push(b as u32);
        adjacency[b].push(a as u32);
    }
    for list in adjacency.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let neighbours = &adjacency[i];
        let k = neighbours.len();
        if k < 2 {
            continue;
        }
        let mut links = 0usize;
        for a in 0..k {
            let na = neighbours[a] as usize;
            for &nb in &neighbours[a + 1..] {
                if adjacency[na].binary_search(&nb).is_ok() {
                    links += 1;
                }
            }
        }
        out[i] = (2.0 * links as f64 / (k as f64 * (k as f64 - 1.0))) as f32;
    }
    out
}

pub fn average_clustering(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|&v| v as f64).sum();
    (sum / values.len() as f64) as f32
}

fn dense_labels<I: Iterator<Item = usize>>(raw: I, n: usize) -> Vec<u32> {
    let mut remap = vec![u32::MAX; n];
    let mut next = 0u32;
    let mut out = Vec::with_capacity(n);
    for root in raw {
        let slot = &mut remap[root.min(n.saturating_sub(1))];
        if *slot == u32::MAX {
            *slot = next;
            next += 1;
        }
        out.push(*slot);
    }
    out
}
