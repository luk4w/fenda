//! Memória do projeto (`.fenda/memory.fnd`): notas com vetor em um arquivo binário somente de acréscimo.
//! Notas não são apagadas; uma nota nova pode substituir outra por meio de uma aresta.
//! O cabeçalho é gravado por último, de modo que uma interrupção não corrompe o arquivo.

use crate::package::{f16_to_f32, f32_to_f16};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn Error>>;

const MAGIC: [u8; 8] = [b'F', b'N', b'D', b'M', b'E', b'M', 0, 1];
const HEADER: usize = 64;
const NODE: usize = 48;
const EDGE: usize = 32;
pub const SUPERSEDES: u32 = 0;

#[derive(Debug, Clone, Default)]
struct Header {
    node_count: u64,
    edge_count: u64,
    last_node: u64,
    dim: u64,
}

impl Header {
    fn bytes(&self) -> [u8; HEADER] {
        let mut b = [0u8; HEADER];
        b[..8].copy_from_slice(&MAGIC);
        b[8..16].copy_from_slice(&self.node_count.to_le_bytes());
        b[16..24].copy_from_slice(&self.edge_count.to_le_bytes());
        b[24..32].copy_from_slice(&self.last_node.to_le_bytes());
        b[32..40].copy_from_slice(&self.dim.to_le_bytes());
        b
    }
}

/// Nota: texto, tipo, data e vetor.
pub struct Note {
    pub id: u64,
    pub title: String,
    pub body: String,
    pub vector: Vec<f32>,
    pub edges: Vec<(u64, u32)>,
}

pub struct Memory {
    path: PathBuf,
    header: Header,
    notes: Vec<Note>,
}

fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

fn read_at(f: &mut File, off: u64, n: usize) -> Option<Vec<u8>> {
    let mut b = vec![0u8; n];
    f.seek(SeekFrom::Start(off)).ok()?;
    f.read_exact(&mut b).ok()?;
    Some(b)
}

fn read_u32(f: &mut File, off: u64) -> Option<u32> {
    read_at(f, off, 4).map(|b| u32::from_le_bytes(b[..4].try_into().unwrap()))
}

/// Localiza `.fenda/memory.fnd` percorrendo os diretórios ancestrais, como o git faz com `.git`.
pub fn find_memory_file(create: bool) -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let target = dir.join(".fenda").join("memory.fnd");
        if target.exists() {
            return Some(target);
        }
        if dir.join(".fenda").exists() || dir.join(".git").exists() {
            if !create {
                return None;
            }
            std::fs::create_dir_all(dir.join(".fenda")).ok()?;
            return Some(target);
        }
        if !dir.pop() {
            return None;
        }
    }
}

impl Memory {
    pub fn open(path: &Path) -> Res<Self> {
        let mut m = Memory {
            path: path.to_path_buf(),
            header: Header::default(),
            notes: Vec::new(),
        };
        m.load()?;
        Ok(m)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.notes.len()
    }

    pub fn find(&self, id: u64) -> Option<&Note> {
        self.notes.iter().find(|n| n.id == id)
    }

    /// Notas que substituem esta.
    pub fn superseded_by(&self, id: u64) -> Vec<u64> {
        self.notes
            .iter()
            .filter(|n| n.edges.iter().any(|&(to, t)| t == SUPERSEDES && to == id))
            .map(|n| n.id)
            .collect()
    }

    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut out: Vec<(u64, f32)> = self
            .notes
            .iter()
            .map(|n| (n.id, n.vector.iter().zip(q).map(|(a, b)| a * b).sum()))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(k);
        out
    }

    /// Acrescenta uma nota e, opcionalmente, a aresta de substituição.
    pub fn add(
        &mut self,
        title: &str,
        body: &str,
        vector: &[f32],
        links: &[(u64, u32)],
    ) -> Res<u64> {
        if self.header.dim != 0 && vector.len() as u64 != self.header.dim {
            return Err("vector dim does not match the memory file".into());
        }
        let mut f = OpenOptions::new().read(true).write(true).open(&self.path)?;
        let mut end = f.seek(SeekFrom::End(0))?;

        let t_off = write_blob(&mut f, &mut end, title.as_bytes())?;
        let b_off = write_blob(&mut f, &mut end, body.as_bytes())?;
        let mut half = Vec::with_capacity(vector.len() * 2);
        for &x in vector {
            half.extend_from_slice(&f32_to_f16(x).to_le_bytes());
        }
        let v_off = write_prefixed(&mut f, &mut end, vector.len() as u32, &half)?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut node = [0u8; NODE];
        node[0..8].copy_from_slice(&self.header.last_node.to_le_bytes());
        node[8..16].copy_from_slice(&t_off.to_le_bytes());
        node[16..24].copy_from_slice(&b_off.to_le_bytes());
        node[24..32].copy_from_slice(&v_off.to_le_bytes());
        node[40..48].copy_from_slice(&created.to_le_bytes());
        let n_off = write_record(&mut f, &mut end, &node)?;

        let mut head = 0u64;
        for &(target, kind) in links {
            let mut e = [0u8; EDGE];
            e[0..8].copy_from_slice(&head.to_le_bytes());
            e[8..16].copy_from_slice(&n_off.to_le_bytes());
            e[16..24].copy_from_slice(&target.to_le_bytes());
            e[24..28].copy_from_slice(&kind.to_le_bytes());
            head = write_record(&mut f, &mut end, &e)?;
        }
        if head != 0 {
            f.seek(SeekFrom::Start(n_off + 32))?;
            f.write_all(&head.to_le_bytes())?;
        }
        f.flush()?;

        self.header.last_node = n_off;
        self.header.node_count += 1;
        self.header.edge_count += links.len() as u64;
        if self.header.dim == 0 {
            self.header.dim = vector.len() as u64;
        }
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&self.header.bytes())?;
        f.flush()?;
        drop(f);

        self.load()?;
        Ok(n_off)
    }

    fn load(&mut self) -> Res<()> {
        self.notes.clear();
        if !self.path.exists() {
            self.header = Header::default();
            File::create(&self.path)?.write_all(&self.header.bytes())?;
            return Ok(());
        }
        let mut f = File::open(&self.path)?;
        let h = read_at(&mut f, 0, HEADER).ok_or("memory file is truncated")?;
        if h[..8] != MAGIC {
            return Err(format!("{} is not a fenda memory file", self.path.display()).into());
        }
        self.header = Header {
            node_count: u64_at(&h, 8),
            edge_count: u64_at(&h, 16),
            last_node: u64_at(&h, 24),
            dim: u64_at(&h, 32),
        };
        let mut off = self.header.last_node;
        while off != 0 {
            let Some(n) = read_at(&mut f, off, NODE) else {
                break;
            };
            let mut note = Note {
                id: off,
                title: read_blob(&mut f, u64_at(&n, 8)),
                body: read_blob(&mut f, u64_at(&n, 16)),
                vector: read_vector(&mut f, u64_at(&n, 24)),
                edges: Vec::new(),
            };
            let mut e = u64_at(&n, 32);
            while e != 0 {
                let Some(edge) = read_at(&mut f, e, EDGE) else {
                    break;
                };
                let kind = u32::from_le_bytes(edge[24..28].try_into().unwrap());
                note.edges.push((u64_at(&edge, 16), kind));
                e = u64_at(&edge, 0);
            }
            self.notes.push(note);
            off = u64_at(&n, 0);
        }
        self.notes.reverse();
        Ok(())
    }
}

fn write_record(f: &mut File, end: &mut u64, bytes: &[u8]) -> Res<u64> {
    let pad = ((8 - *end % 8) % 8) as usize;
    f.seek(SeekFrom::Start(*end))?;
    f.write_all(&vec![0u8; pad])?;
    *end += pad as u64;
    let off = *end;
    f.write_all(bytes)?;
    *end += bytes.len() as u64;
    Ok(off)
}

fn write_prefixed(f: &mut File, end: &mut u64, n: u32, bytes: &[u8]) -> Res<u64> {
    let off = write_record(f, end, &n.to_le_bytes())?;
    f.write_all(bytes)?;
    *end += bytes.len() as u64;
    Ok(off)
}

fn write_blob(f: &mut File, end: &mut u64, s: &[u8]) -> Res<u64> {
    write_prefixed(f, end, s.len() as u32, s)
}

fn read_blob(f: &mut File, off: u64) -> String {
    match read_u32(f, off) {
        Some(n) if n <= 1 << 28 => read_at(f, off + 4, n as usize)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn read_vector(f: &mut File, off: u64) -> Vec<f32> {
    match read_u32(f, off) {
        Some(n) if n <= 1 << 20 => read_at(f, off + 4, n as usize * 2)
            .map(|b| {
                b.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}
