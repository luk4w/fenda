//! Infraestrutura comum a embedding e reordenador: localização do modelo, leitura do `fenda.json`, sessão do ONNX Runtime e execução em lotes.

use ort::environment::Environment;
#[cfg(feature = "gpu")]
use ort::ep;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde::Deserialize;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tokenizers::{Encoding, Tokenizer, TruncationParams, TruncationStrategy};

pub type Res<T> = Result<T, Box<dyn Error>>;

// Variantes no diretório do modelo: fp16 para GPU, int8 para CPU.
const GPU_FILE: &str = "model.onnx";
const CPU_FILE: &str = "model.int8.onnx";

// Tamanho do lote, em tokens.
const BATCH_TOKENS: usize = 16 * 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::Gpu => "GPU (DirectML)",
            Device::Cpu => "CPU",
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Config {
    pub kind: String,
    pub output: String,
    #[serde(default)]
    pub pooling: Pooling,
    pub max_tokens: usize,
    #[serde(default)]
    pub query_prefix: String,
    #[serde(default)]
    pub passage_prefix: String,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Pooling {
    #[default]
    None,
    Cls,
    Mean,
    Last,
}

pub fn folder_name(name: &str) -> &str {
    let n = name.rsplit('/').next().unwrap_or(name);
    n.split(':').next().unwrap_or(n)
}

fn roots(dir: Option<&Path>) -> Vec<PathBuf> {
    if let Some(d) = dir {
        return vec![d.to_path_buf()];
    }
    let mut out = Vec::new();
    if let Some(d) = std::env::var_os("FENDA_MODELS") {
        out.push(PathBuf::from(d));
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(p) = cwd
            .ancestors()
            .map(|a| a.join(".fenda").join("models"))
            .find(|p| p.is_dir())
        {
            out.push(p);
        }
    }
    if let Some(h) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        out.push(PathBuf::from(h).join(".fenda").join("models"));
    }
    out
}

/// Localiza o diretório do modelo: `--models`, `$FENDA_MODELS`, `.fenda/models`, `~/.fenda/models`.
pub fn find_model(name: &str, dir: Option<&Path>) -> Res<PathBuf> {
    let folder = folder_name(name);
    let places = roots(dir);
    for r in &places {
        let p = r.join(folder);
        if p.join(GPU_FILE).is_file() || p.join(CPU_FILE).is_file() {
            return Ok(p);
        }
    }
    let looked: Vec<String> = places.iter().map(|p| p.display().to_string()).collect();
    Err(format!(
        "model '{folder}' not found; looked for {folder}/{GPU_FILE} or {folder}/{CPU_FILE} in: {}",
        if looked.is_empty() {
            "(nowhere)".to_string()
        } else {
            looked.join(", ")
        }
    )
    .into())
}

pub(crate) struct Loaded {
    pub name: String,
    pub device: Device,
    pub config: Config,
    pub tokenizer: Tokenizer,
    pad: u32,
    token_types: bool,
    session: Mutex<Session>,
}

pub(crate) fn load(name: &str, dir: Option<&Path>, device: Device, kind: &str) -> Res<Loaded> {
    let path = find_model(name, dir)?;
    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(path.join("fenda.json"))
            .map_err(|e| format!("{}: fenda.json: {e}", path.display()))?,
    )
    .map_err(|e| format!("{}: fenda.json: {e}", path.display()))?;
    if config.kind != kind {
        return Err(format!("{} is a {} model, not {kind}", path.display(), config.kind).into());
    }

    let mut tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
        .map_err(|e| format!("{}: tokenizer.json: {e}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: config.max_tokens,
            strategy: TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    tokenizer.with_padding(None);
    let pad = tokenizer
        .token_to_id("<pad>")
        .or_else(|| tokenizer.token_to_id("[PAD]"))
        .unwrap_or(0);

    let (session, device) = open_session(&path, device)?;
    let token_types = session
        .inputs()
        .iter()
        .any(|i| i.name() == "token_type_ids");
    Ok(Loaded {
        name: folder_name(name).to_string(),
        device,
        config,
        tokenizer,
        pad,
        token_types,
        session: Mutex::new(session),
    })
}

pub fn open_session(path: &Path, device: Device) -> Res<(Session, Device)> {
    let (gpu, cpu) = (path.join(GPU_FILE), path.join(CPU_FILE));
    if device == Device::Gpu && gpu.is_file() {
        match build(&gpu, true) {
            Ok(s) => return Ok((s, Device::Gpu)),
            Err(e) => eprintln!("[models] no GPU for {} ({e}); using the CPU", gpu.display()),
        }
    }
    Ok((
        build(if cpu.is_file() { &cpu } else { &gpu }, false)?,
        Device::Cpu,
    ))
}

fn environment() -> Res<&'static Environment> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("fenda")
            .build()
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| e.clone().into())
}

// GPU disponível apenas com o recurso `gpu` (DirectML); o leitor usa CPU.
fn build(model: &Path, gpu: bool) -> Res<Session> {
    let mut b = Session::builder(environment()?)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    if gpu {
        #[cfg(feature = "gpu")]
        {
            b = b
                .with_memory_pattern(false)?
                .with_parallel_execution(false)?
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])?;
        }
        #[cfg(not(feature = "gpu"))]
        return Err("this build has no GPU support".into());
    }
    Ok(b.commit_from_file(model)?)
}

#[derive(Clone)]
pub(crate) struct Row {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
    pub real: usize,
}

impl Loaded {
    pub fn run(&self, encodings: &[Encoding]) -> Res<Vec<Row>> {
        let mut session = self.session.lock().map_err(|_| "model session poisoned")?;
        let mut out: Vec<Option<Row>> = vec![None; encodings.len()];
        for batch in self.batches(encodings) {
            let part: Vec<&Encoding> = batch.iter().map(|&i| &encodings[i]).collect();
            for (&i, row) in batch.iter().zip(self.run_batch(&mut session, &part)?) {
                out[i] = Some(row);
            }
        }
        Ok(out
            .into_iter()
            .map(|r| r.expect("every input has an output"))
            .collect())
    }

    fn batches(&self, encodings: &[Encoding]) -> Vec<Vec<usize>> {
        if self.device == Device::Cpu {
            return (0..encodings.len()).map(|i| vec![i]).collect();
        }
        let len = |i: usize| encodings[i].get_ids().len();
        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| len(i));
        let mut out = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && (end + 1 - start) * len(order[end]) <= BATCH_TOKENS {
                end += 1;
            }
            out.push(order[start..end].to_vec());
            start = end;
        }
        out
    }

    fn run_batch(&self, session: &mut Session, part: &[&Encoding]) -> Res<Vec<Row>> {
        let rows = part.len();
        let cols = part.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let mut ids = vec![self.pad as i64; rows * cols];
        let mut mask = vec![0i64; rows * cols];
        let mut types = vec![0i64; rows * cols];
        for (r, e) in part.iter().enumerate() {
            for (c, (&id, &t)) in e.get_ids().iter().zip(e.get_type_ids()).enumerate() {
                ids[r * cols + c] = id as i64;
                mask[r * cols + c] = 1;
                types[r * cols + c] = t as i64;
            }
        }
        let shape = [rows, cols];
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((shape, ids))?,
            "attention_mask" => Tensor::from_array((shape, mask))?,
        ];
        if self.token_types {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array((shape, types))?.into(),
            ));
        }
        let outputs = session.run(inputs)?;
        let (shape, data) = outputs[self.config.output.as_str()].try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims.first() != Some(&rows) {
            return Err(format!(
                "{}: output '{}' has shape {dims:?}, expected {rows} rows",
                self.name, self.config.output
            )
            .into());
        }
        let each = data.len() / rows;
        Ok((0..rows)
            .map(|r| Row {
                dims: dims[1..].to_vec(),
                data: data[r * each..(r + 1) * each].to_vec(),
                real: part[r].get_ids().len(),
            })
            .collect())
    }
}
