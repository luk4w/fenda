//! Pipeline de busca: consulta -> vetor -> `POOL` melhores por cosseno -> reordenador -> k resultados.

use crate::runtime::{Device, Res, folder_name};
use crate::{Embedder, Reranker};
use fnd::multi::{self, Doc, MultiHit};
use std::path::Path;

// Quantidade de trechos reavaliados pelo reordenador.
const POOL: usize = 30;

pub struct Models {
    pub embedder: Embedder,
    pub reranker: Reranker,
}

impl Models {
    pub fn load(embed_model: &str, models_dir: Option<&Path>, device: Device) -> Res<Self> {
        let embedder = Embedder::load(embed_model, models_dir, device)?;
        eprintln!(
            "[models] embedding {} on {}",
            embedder.name(),
            embedder.device().name()
        );
        let reranker = Reranker::load(models_dir, device)?;
        eprintln!(
            "[models] reranker {} on {}",
            reranker.name(),
            reranker.device().name()
        );
        Ok(Models { embedder, reranker })
    }
}

pub fn search(
    docs: &[Doc],
    selected: &[usize],
    models: &Models,
    query: &str,
    k: usize,
) -> Res<(Vec<f32>, Vec<MultiHit>)> {
    let model = multi::common_model(docs, selected)?;
    if folder_name(model) != models.embedder.name() {
        return Err(format!(
            "the packages were compiled with the embedding model {model}, but {} is loaded; vectors of different models do not compare",
            models.embedder.name()
        )
        .into());
    }
    let q = models
        .embedder
        .embed_queries(&[query.to_string()])?
        .remove(0);
    let mut hits = multi::search(docs, selected, &q, k.max(POOL))?;
    let passages: Vec<String> = hits
        .iter()
        .map(|h| {
            let t = &docs[h.doc].pkg.tree;
            t.embedding_text(&t.chunks[h.chunk])
        })
        .collect();
    for (h, s) in hits
        .iter_mut()
        .zip(models.reranker.score(query, &passages)?)
    {
        h.rerank = Some(s);
    }
    hits.sort_by(|a, b| {
        b.rerank
            .unwrap_or(f32::MIN)
            .total_cmp(&a.rerank.unwrap_or(f32::MIN))
    });
    hits.truncate(k);
    Ok((q, hits))
}
