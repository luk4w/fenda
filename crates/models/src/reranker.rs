//! Reordenador (cross-encoder): avalia consulta e trecho em conjunto. Por ser custoso, é aplicado apenas aos primeiros resultados da busca vetorial.

use crate::runtime::{self, Device, Loaded, Res};
use std::path::Path;

pub const MODEL: &str = "mmarco-mMiniLMv2-L12-H384-v1";

pub struct Reranker {
    m: Loaded,
}

impl Reranker {
    pub fn load(models_dir: Option<&Path>, device: Device) -> Res<Self> {
        Ok(Reranker {
            m: runtime::load(MODEL, models_dir, device, "reranker")?,
        })
    }

    pub fn name(&self) -> &str {
        &self.m.name
    }

    pub fn device(&self) -> Device {
        self.m.device
    }

    pub fn score(&self, query: &str, passages: &[String]) -> Res<Vec<f32>> {
        let pairs: Vec<(String, String)> = passages
            .iter()
            .map(|p| (query.to_string(), p.clone()))
            .collect();
        let encodings = self
            .m
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| format!("{}: tokenizer: {e}", self.m.name))?;
        self.m
            .run(&encodings)?
            .into_iter()
            .map(|row| match row.data.first() {
                Some(&s) if row.data.len() == 1 => Ok(s),
                _ => Err(format!(
                    "{}: expected one logit per pair, got {:?}",
                    self.m.name, row.dims
                )
                .into()),
            })
            .collect()
    }
}
