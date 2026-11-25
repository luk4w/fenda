//! Embedding: um vetor normalizado por texto; o mesmo modelo atende trechos e consultas.

use crate::runtime::{self, Device, Loaded, Pooling, Res, Row};
use std::path::Path;

pub struct Embedder {
    m: Loaded,
}

impl Embedder {
    pub fn load(name: &str, models_dir: Option<&Path>, device: Device) -> Res<Self> {
        Ok(Embedder {
            m: runtime::load(name, models_dir, device, "embedding")?,
        })
    }

    pub fn name(&self) -> &str {
        &self.m.name
    }

    pub fn device(&self) -> Device {
        self.m.device
    }

    pub fn embed_queries(&self, texts: &[String]) -> Res<Vec<Vec<f32>>> {
        self.embed(texts, &self.m.config.query_prefix)
    }

    pub fn embed_passages(&self, texts: &[String]) -> Res<Vec<Vec<f32>>> {
        self.embed(texts, &self.m.config.passage_prefix)
    }

    fn embed(&self, texts: &[String], prefix: &str) -> Res<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let encodings = self
            .m
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| format!("{}: tokenizer: {e}", self.m.name))?;
        self.m
            .run(&encodings)?
            .into_iter()
            .map(|row| {
                let mut v = self.pool(row)?;
                normalize(&mut v);
                Ok(v)
            })
            .collect()
    }

    fn pool(&self, row: Row) -> Res<Vec<f32>> {
        let pooling = self.m.config.pooling;
        if let Pooling::None = pooling {
            if row.dims.len() != 1 {
                return Err(format!(
                    "{}: output is per token ({:?}); set \"pooling\" in fenda.json",
                    self.m.name, row.dims
                )
                .into());
            }
            return Ok(row.data);
        }
        if row.dims.len() != 2 {
            return Err(format!(
                "{}: pooling needs a per-token output, got {:?}",
                self.m.name, row.dims
            )
            .into());
        }
        let dim = row.dims[1];
        let token = |t: usize| &row.data[t * dim..(t + 1) * dim];
        Ok(match pooling {
            Pooling::Cls => token(0).to_vec(),
            Pooling::Last => token(row.real.max(1) - 1).to_vec(),
            Pooling::Mean => {
                let mut sum = vec![0f32; dim];
                for t in 0..row.real {
                    for (s, x) in sum.iter_mut().zip(token(t)) {
                        *s += x;
                    }
                }
                sum.iter().map(|s| s / row.real.max(1) as f32).collect()
            }
            Pooling::None => unreachable!(),
        })
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}
