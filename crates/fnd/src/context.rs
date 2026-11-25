//! Montagem do contexto do `ask`: a partir das seções recuperadas, expande um nível pelo sumário (pai, irmãs, filhas) e pelas referências cruzadas, e seleciona o trecho mais relevante de cada seção.

use crate::package::{Hit, Tensor};
use crate::tree::{Edge, Tree};
use std::collections::BTreeMap;

// Peso de cada relação na pontuação da seção.
const W_SELF: f64 = 1.0;
const W_PARENT: f64 = 0.7;
const W_SIBLING: f64 = 0.6;
const W_CHILD: f64 = 0.6;
const W_REFERENCE: f64 = 0.4;

pub fn expand_context(
    tree: &Tree,
    edges: &[Edge],
    vectors: &Tensor,
    query: &[f32],
    hits: &[Hit],
) -> Vec<usize> {
    let mut children: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for s in &tree.sections {
        if s.id != 0 {
            children.entry(s.parent).or_default().push(s.id);
        }
    }
    let mut linked: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for e in edges {
        linked.entry(e.from).or_default().push(e.to);
        linked.entry(e.to).or_default().push(e.from);
    }

    let mut entry: BTreeMap<i32, f64> = BTreeMap::new();
    for h in hits {
        let s = tree.chunks[h.chunk].section;
        let best = entry.entry(s).or_insert(f64::MIN);
        *best = best.max(h.score as f64);
    }

    let mut score: BTreeMap<i32, f64> = BTreeMap::new();
    for (&s, &w) in &entry {
        *score.entry(s).or_default() += w * W_SELF;
        let parent = tree.sections[s as usize].parent;
        if parent > 0 {
            *score.entry(parent).or_default() += w * W_PARENT;
            for &sibling in children.get(&parent).map_or(&[][..], Vec::as_slice) {
                if sibling != s {
                    *score.entry(sibling).or_default() += w * W_SIBLING;
                }
            }
        }
        for &child in children.get(&s).map_or(&[][..], Vec::as_slice) {
            *score.entry(child).or_default() += w * W_CHILD;
        }
        for &other in linked.get(&s).map_or(&[][..], Vec::as_slice) {
            *score.entry(other).or_default() += w * W_REFERENCE;
        }
    }

    let mut best: BTreeMap<i32, (f32, usize)> = BTreeMap::new();
    for (i, c) in tree.chunks.iter().enumerate() {
        if !score.contains_key(&c.section) || i >= vectors.rows {
            continue;
        }
        let dot: f32 = vectors.row(i).iter().zip(query).map(|(a, b)| a * b).sum();
        let slot = best.entry(c.section).or_insert((f32::MIN, i));
        if dot > slot.0 {
            *slot = (dot, i);
        }
    }

    let mut ranked: Vec<(bool, f64, i32)> = score
        .iter()
        .filter(|(s, _)| best.contains_key(s))
        .map(|(&s, &w)| (!entry.contains_key(&s), w, s))
        .collect();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
    let mut out = Vec::new();
    for &(is_neighbor, _, sec) in &ranked {
        if !is_neighbor {
            for (i, c) in tree.chunks.iter().enumerate() {
                if c.section == sec && i < vectors.rows {
                    out.push(i);
                }
            }
        } else {
            out.push(best[&sec].1);
        }
    }
    out
}
