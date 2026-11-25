//! Busca em vários livros com ranking único.
//! Todos os livros devem usar o mesmo modelo de embedding, pois pontuações de modelos distintos não são comparáveis.

use crate::package::{self, Hit, Package};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

type Res<T> = Result<T, Box<dyn Error>>;

/// Livro carregado.
pub struct Doc {
    pub name: String,
    pub model: String,
    pub pkg: Package,
    pub path: PathBuf,
    pub modified: Option<SystemTime>,
}

pub struct MultiHit {
    pub doc: usize,
    pub chunk: usize,
    pub score: f32,
    pub rerank: Option<f32>,
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl Doc {
    /// Recarrega o livro se o arquivo foi modificado.
    pub fn refresh(&mut self) -> Res<()> {
        let now = modified(&self.path);
        if now != self.modified {
            self.pkg = package::read_package(&self.path)?;
            self.modified = now;
        }
        Ok(())
    }

    pub fn replace(&mut self, pkg: Package) -> Res<()> {
        self.pkg = pkg;
        self.modified = modified(&self.path);
        Ok(())
    }

    pub fn section_tokens(&self, sec: i32) -> i32 {
        self.pkg
            .tree
            .chunks
            .iter()
            .filter(|c| c.section == sec)
            .map(|c| c.tokens_est)
            .sum()
    }

    pub fn first_page(&self, sec: i32) -> i32 {
        self.pkg
            .tree
            .chunks
            .iter()
            .filter(|c| c.section == sec && c.page_first > 0)
            .map(|c| c.page_first)
            .min()
            .unwrap_or(0)
    }
}

/// Nome do livro: nome do arquivo sem a extensão `.fnd`.
pub fn short_name(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

pub fn load(path: &str) -> Res<Doc> {
    let pkg = package::read_package(Path::new(path))?;
    if pkg.embed_model.is_empty() {
        return Err(format!("{path} has no embeddings; recompile it").into());
    }
    let model = pkg.embed_model.clone();
    Ok(Doc {
        name: short_name(path),
        model,
        pkg,
        path: PathBuf::from(path),
        modified: modified(Path::new(path)),
    })
}

/// Modelo de embedding comum aos livros selecionados; erro se houver mais de um.
pub fn common_model<'a>(docs: &'a [Doc], selected: &[usize]) -> Res<&'a str> {
    let Some(&first) = selected.first() else {
        return Err("no document to search".into());
    };
    let model = &docs[first].model;
    if let Some(&other) = selected.iter().find(|&&i| docs[i].model != *model) {
        return Err(format!(
            "{} uses the embedding model {} and {} uses {}; scores of different models do not compare, so search them separately",
            docs[first].name, model, docs[other].name, docs[other].model
        )
        .into());
    }
    Ok(model)
}

/// Os k trechos mais relevantes entre todos os livros.
pub fn search(docs: &[Doc], selected: &[usize], query: &[f32], k: usize) -> Res<Vec<MultiHit>> {
    common_model(docs, selected)?;
    let mut all = Vec::new();
    for &i in selected {
        let t = &docs[i].pkg.tree;
        for h in package::top_k(&docs[i].pkg.vectors, query, k, |c| t.chunks[c].is_body())? {
            all.push(MultiHit {
                doc: i,
                chunk: h.chunk,
                score: h.score,
                rerank: None,
            });
        }
    }
    all.sort_by(|a, b| b.score.total_cmp(&a.score));
    all.truncate(k);
    Ok(all)
}

pub fn hits_of(hits: &[MultiHit], doc: usize) -> Vec<Hit> {
    hits.iter()
        .filter(|h| h.doc == doc)
        .map(|h| Hit {
            chunk: h.chunk,
            score: h.score,
        })
        .collect()
}

pub fn docs_in_order(hits: &[MultiHit]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for h in hits {
        if !out.contains(&h.doc) {
            out.push(h.doc);
        }
    }
    out
}

pub fn split_inputs(inputs: &[String]) -> Res<(&[String], &str)> {
    match inputs.split_last() {
        Some((query, files)) if !files.is_empty() => Ok((files, query.as_str())),
        _ => Err("give one or more .fnd files followed by the query".into()),
    }
}
