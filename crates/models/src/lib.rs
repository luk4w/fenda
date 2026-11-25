//! Modelos executados no próprio processo via ONNX Runtime, na CPU: embedding (consulta -> vetor) e reordenador (reavaliação dos melhores trechos).
//!
//! Cada modelo é um diretório com `model.int8.onnx`, `tokenizer.json` e `fenda.json`, que descreve como interpretar a saída.

pub mod embedder;
pub mod reranker;
pub mod retrieve;
mod runtime;

pub use embedder::Embedder;
pub use reranker::Reranker;
pub use retrieve::Models;
pub use runtime::{Device, find_model, folder_name, open_session};
