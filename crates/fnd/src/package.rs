//! Pacote `.fnd`: banco SQLite em arquivo único.
//!
//!   info          origem, hash do PDF e modelo de embedding
//!   pages         dimensões de cada página
//!   blocks        conteúdo reconhecido nas páginas (imutável, protegido por trigger)
//!   sections      o sumário; a raiz contém o nome do livro
//!   chunks        trechos de cada seção
//!   chunk_blocks  blocos de cada trecho, em ordem
//!   vectors       vetor f16 por texto de trecho
//!   floats        figura/tabela -> seção da legenda
//!   edges         referências cruzadas entre seções
//!   edits         correções do sumário (fragmento, papel, data)

use crate::edits::{Edit, Edits, Role, piece_id, split_piece};
use crate::parser::ParserOptions;
use crate::tree::{Block, Edge, Section, Tree};
use rusqlite::{Connection, OpenFlags, Transaction, TransactionBehavior, params};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::path::Path;

type Res<T> = Result<T, Box<dyn Error>>;

/// Conversão para f16, formato de armazenamento dos vetores.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let raw_exp = (x >> 23) & 0xFF;
    let exp = raw_exp as i32 - 127 + 15;
    let mut mant = x & 0x7F_FFFF;

    if raw_exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut m = mant >> shift;
        let rest = mant & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rest > half || (rest == half && (m & 1) != 0) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut m = mant >> 13;
    let rest = mant & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && (m & 1) != 0) {
        m += 1;
    }
    ((sign as u32 | ((exp as u32) << 10)) + m) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1F) as i32;
    let mut mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 1;
            while mant & 0x400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x3FF;
            sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[derive(Debug, Clone, Default)]
pub struct Tensor {
    pub rows: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn from_rows(rows: Vec<Vec<f32>>) -> Res<Tensor> {
        let dim = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != dim) {
            return Err("vectors with different dimensions".into());
        }
        Ok(Tensor {
            rows: rows.len(),
            dim,
            data: rows.into_iter().flatten().collect(),
        })
    }
}

/// Pacote completo em memória.
pub struct Package {
    pub embed_model: String,
    /// SHA-256 do PDF compilado; ausente em pacotes anteriores a este campo.
    pub source_sha256: Option<String>,
    pub chunking: ParserOptions,
    pub blocks: Vec<Block>,
    pub pages: BTreeMap<i32, (i32, i32)>,
    pub tree: Tree,
    pub edits: Edits,
    pub vectors: Tensor,
    pub edges: Vec<Edge>,
}

const APPLICATION_ID: i32 = 0x464E_4431;
const SCHEMA_VERSION: i32 = 1;

const SCHEMA: &str = "
CREATE TABLE info(
    source TEXT NOT NULL,
    generator TEXT NOT NULL,
    embed_model TEXT,
    target_tokens INTEGER NOT NULL,
    max_tokens INTEGER NOT NULL,
    source_sha256 TEXT) STRICT;
CREATE TABLE pages(page INTEGER PRIMARY KEY, width INTEGER NOT NULL, height INTEGER NOT NULL) STRICT;
CREATE TABLE blocks(
    seq INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    page INTEGER NOT NULL REFERENCES pages(page),
    label TEXT NOT NULL,
    text TEXT NOT NULL,
    caption TEXT NOT NULL,
    tag TEXT NOT NULL,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    score REAL NOT NULL) STRICT;
CREATE TABLE sections(
    id INTEGER PRIMARY KEY,
    parent INTEGER REFERENCES sections(id),
    level INTEGER NOT NULL,
    title TEXT NOT NULL,
    block TEXT REFERENCES blocks(id),
    start INTEGER,
    UNIQUE(block, start)) STRICT;
CREATE TABLE vectors(key INTEGER PRIMARY KEY, vector BLOB NOT NULL) STRICT;
CREATE TABLE chunks(
    id INTEGER PRIMARY KEY,
    section INTEGER NOT NULL REFERENCES sections(id),
    vector INTEGER REFERENCES vectors(key)) STRICT;
CREATE TABLE chunk_blocks(
    chunk INTEGER NOT NULL REFERENCES chunks(id),
    pos INTEGER NOT NULL,
    block TEXT NOT NULL REFERENCES blocks(id),
    start INTEGER NOT NULL,
    PRIMARY KEY(chunk, pos)) WITHOUT ROWID, STRICT;
CREATE TABLE floats(label TEXT PRIMARY KEY, section INTEGER NOT NULL REFERENCES sections(id)) STRICT;
CREATE TABLE edges(
    type TEXT NOT NULL,
    source INTEGER NOT NULL REFERENCES sections(id),
    target INTEGER NOT NULL REFERENCES sections(id),
    label TEXT NOT NULL,
    UNIQUE(type, source, target, label)) STRICT;
CREATE TABLE edits(
    block TEXT NOT NULL REFERENCES blocks(id),
    start INTEGER NOT NULL,
    stop INTEGER CHECK(stop IS NULL OR stop > start),
    role TEXT NOT NULL CHECK(role IN ('chapter', 'section', 'subsection', 'heading', 'text', 'title', 'join')),
    at TEXT NOT NULL,
    PRIMARY KEY(block, start)) WITHOUT ROWID, STRICT;
CREATE TRIGGER info_fixed BEFORE UPDATE ON info BEGIN SELECT RAISE(ABORT, 'info is fixed'); END;
CREATE TRIGGER info_kept BEFORE DELETE ON info BEGIN SELECT RAISE(ABORT, 'info is fixed'); END;
CREATE TRIGGER pages_fixed BEFORE UPDATE ON pages BEGIN SELECT RAISE(ABORT, 'pages are fixed'); END;
CREATE TRIGGER pages_kept BEFORE DELETE ON pages BEGIN SELECT RAISE(ABORT, 'pages are fixed'); END;
CREATE TRIGGER blocks_fixed BEFORE UPDATE ON blocks BEGIN SELECT RAISE(ABORT, 'blocks are fixed: what the vision read is never changed'); END;
CREATE TRIGGER blocks_kept BEFORE DELETE ON blocks BEGIN SELECT RAISE(ABORT, 'blocks are fixed: what the vision read is never changed'); END;
";

/// Chave do vetor: hash FNV-1a do texto submetido ao modelo. Textos idênticos compartilham o vetor, e uma correção recalcula apenas os novos.
pub fn vector_key(embedding_text: &str) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in embedding_text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h as i64
}

fn sql(e: rusqlite::Error) -> Box<dyn Error> {
    format!("sqlite: {e}").into()
}

fn vector_blob(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_f16(x).to_le_bytes())
        .collect()
}

fn put_tree(tx: &Transaction, p: &Package) -> Res<()> {
    let t = &p.tree;
    let mut s = tx
        .prepare("INSERT INTO sections VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
        .map_err(sql)?;
    for x in &t.sections {
        let parent = (x.parent >= 0).then_some(x.parent);
        let (block, start) = if x.block.is_empty() {
            (None, None)
        } else {
            let (b, s) = split_piece(&x.block);
            (Some(b), Some(s as i64))
        };
        s.execute(params![x.id, parent, x.level, x.title, block, start])
            .map_err(sql)?;
    }
    let keys: Vec<Option<i64>> = t
        .chunks
        .iter()
        .map(|ch| (p.vectors.rows > 0).then(|| vector_key(&t.embedding_text(ch))))
        .collect();
    let mut order: Vec<usize> = (0..keys.len()).filter(|&i| keys[i].is_some()).collect();
    order.sort_by_key(|&i| keys[i]);
    let mut v = tx
        .prepare("INSERT OR IGNORE INTO vectors VALUES (?1, ?2)")
        .map_err(sql)?;
    for i in order {
        v.execute(params![keys[i], vector_blob(p.vectors.row(i))])
            .map_err(sql)?;
    }
    let mut c = tx
        .prepare("INSERT INTO chunks VALUES (?1, ?2, ?3)")
        .map_err(sql)?;
    let mut cb = tx
        .prepare("INSERT INTO chunk_blocks VALUES (?1, ?2, ?3, ?4)")
        .map_err(sql)?;
    for (i, ch) in t.chunks.iter().enumerate() {
        c.execute(params![i as i64, ch.section, keys[i]])
            .map_err(sql)?;
        for (pos, b) in ch.blocks.iter().enumerate() {
            let (b, s) = split_piece(b);
            cb.execute(params![i as i64, pos as i64, b, s as i64])
                .map_err(sql)?;
        }
    }
    let mut f = tx
        .prepare("INSERT INTO floats VALUES (?1, ?2)")
        .map_err(sql)?;
    for (k, v) in &t.floats {
        f.execute(params![k, v]).map_err(sql)?;
    }
    let mut e = tx
        .prepare("INSERT INTO edges VALUES (?1, ?2, ?3, ?4)")
        .map_err(sql)?;
    for x in &p.edges {
        e.execute(params![x.etype, x.from, x.to, x.label])
            .map_err(sql)?;
    }
    let mut d = tx
        .prepare("INSERT INTO edits VALUES (?1, ?2, ?3, ?4, ?5)")
        .map_err(sql)?;
    for x in &p.edits.0 {
        d.execute(params![
            x.block,
            x.start as i64,
            x.stop.map(|n| n as i64),
            x.role.name(),
            x.at
        ])
        .map_err(sql)?;
    }
    Ok(())
}

/// Grava o pacote completo.
pub fn write_package(path: &Path, p: &Package) -> Res<()> {
    let partial = path.with_extension("fnd.partial");
    let _ = std::fs::remove_file(&partial);
    {
        let mut db = Connection::open(&partial).map_err(sql)?;
        db.execute_batch(&format!(
            "PRAGMA page_size = 16384; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;
             PRAGMA application_id = {APPLICATION_ID}; PRAGMA user_version = {SCHEMA_VERSION};"
        ))
        .map_err(sql)?;
        db.execute_batch(SCHEMA).map_err(sql)?;
        let tx = db.transaction().map_err(sql)?;
        {
            let model = (!p.embed_model.is_empty()).then_some(&p.embed_model);
            tx.execute(
                "INSERT INTO info VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    p.tree.source,
                    concat!("fenda ", env!("CARGO_PKG_VERSION")),
                    model,
                    p.chunking.target_tokens,
                    p.chunking.max_tokens,
                    p.source_sha256
                ],
            )
            .map_err(sql)?;
            let mut pg = tx
                .prepare("INSERT INTO pages VALUES (?1, ?2, ?3)")
                .map_err(sql)?;
            for (k, (w, h)) in &p.pages {
                pg.execute(params![k, w, h]).map_err(sql)?;
            }
            let mut b = tx
                .prepare(
                    "INSERT INTO blocks VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .map_err(sql)?;
            for (i, x) in p.blocks.iter().enumerate() {
                let bb: [Option<i32>; 4] = match x.bbox.as_slice() {
                    [a, b, c, d] => [Some(*a), Some(*b), Some(*c), Some(*d)],
                    [] => [None; 4],
                    _ => {
                        return Err(
                            format!("block {}: bbox with {} values", x.id, x.bbox.len()).into()
                        );
                    }
                };
                b.execute(params![
                    i as i64, x.id, x.page, x.label, x.text, x.caption, x.tag, bb[0], bb[1], bb[2],
                    bb[3], x.score
                ])
                .map_err(sql)?;
            }
        }
        put_tree(&tx, p)?;
        tx.commit().map_err(sql)?;
    }
    std::fs::rename(&partial, path)?;
    Ok(())
}

type EditRow = (String, i64, Option<i64>, String, String);

fn edit_rows(db: &Connection) -> Res<Vec<EditRow>> {
    let mut q = db
        .prepare("SELECT block, start, stop, role, at FROM edits ORDER BY block, start")
        .map_err(sql)?;
    let rows = q
        .query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })
        .map_err(sql)?;
    Ok(rows.collect::<Result<_, _>>().map_err(sql)?)
}

/// Grava, em uma transação, apenas o que a correção altera. Recusa a gravação se outra sessão modificou as edições depois de `base`.
pub fn save_tree(path: &Path, base: &Edits, p: &Package) -> Res<()> {
    let mut db = open(path, false)?;
    db.execute_batch("PRAGMA foreign_keys = ON;").map_err(sql)?;
    let tx = db
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(sql)?;
    let mut expected: Vec<EditRow> = base
        .0
        .iter()
        .map(|e| {
            (
                e.block.clone(),
                e.start as i64,
                e.stop.map(|n| n as i64),
                e.role.name().to_string(),
                e.at.clone(),
            )
        })
        .collect();
    expected.sort();
    if edit_rows(&tx)? != expected {
        return Err("another session changed this book a moment ago; call again".into());
    }
    tx.execute_batch(
        "DELETE FROM chunk_blocks; DELETE FROM chunks; DELETE FROM floats; DELETE FROM edges; DELETE FROM edits; DELETE FROM sections;",
    )
    .map_err(sql)?;
    put_tree(&tx, p)?;
    tx.execute(
        "DELETE FROM vectors WHERE key NOT IN (SELECT vector FROM chunks WHERE vector IS NOT NULL)",
        [],
    )
    .map_err(sql)?;
    tx.commit().map_err(sql)?;
    Ok(())
}

fn open(path: &Path, read_only: bool) -> Res<Connection> {
    let flags = if read_only {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    } else {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    };
    let db = Connection::open_with_flags(path, flags).map_err(sql)?;
    let not_ours = |e: String| {
        format!(
            "{} is not a fenda package ({e}); compile it again",
            path.display()
        )
    };
    let app: i32 = db
        .query_row("PRAGMA application_id", [], |r| r.get(0))
        .map_err(|e| not_ours(e.to_string()))?;
    if app != APPLICATION_ID {
        return Err(not_ours(format!("application_id {app:#x}")).into());
    }
    let version: i32 = db
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(sql)?;
    if version != SCHEMA_VERSION {
        return Err(format!(
            "{} has schema version {version}, this fenda reads {SCHEMA_VERSION}; compile it again",
            path.display()
        )
        .into());
    }
    Ok(db)
}

/// Lê o pacote; o texto dos trechos é reconstruído a partir dos blocos.
pub fn read_package(path: &Path) -> Res<Package> {
    let db = open(path, true)?;
    let (source, embed_model, target_tokens, max_tokens): (String, Option<String>, i32, i32) = db
        .query_row(
            "SELECT source, embed_model, target_tokens, max_tokens FROM info",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .map_err(sql)?;
    // Pacotes anteriores à coluna não têm o hash do PDF.
    let source_sha256: Option<String> = db
        .query_row("SELECT source_sha256 FROM info", [], |r| r.get(0))
        .unwrap_or(None);

    let mut pages = BTreeMap::new();
    let mut q = db
        .prepare("SELECT page, width, height FROM pages")
        .map_err(sql)?;
    for r in q
        .query_map([], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))
        .map_err(sql)?
    {
        let (k, v) = r.map_err(sql)?;
        pages.insert(k, v);
    }

    let mut q = db.prepare("SELECT id, page, label, text, caption, tag, x0, y0, x1, y1, score FROM blocks ORDER BY seq").map_err(sql)?;
    let blocks: Vec<Block> = q
        .query_map([], |r| {
            let bb: [Option<i32>; 4] = [r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?];
            Ok(Block {
                id: r.get(0)?,
                page: r.get(1)?,
                label: r.get(2)?,
                text: r.get(3)?,
                caption: r.get(4)?,
                tag: r.get(5)?,
                bbox: bb.iter().flatten().copied().collect(),
                score: r.get(10)?,
                ..Default::default()
            })
        })
        .map_err(sql)?
        .collect::<Result<_, _>>()
        .map_err(sql)?;

    let mut q = db
        .prepare("SELECT id, parent, level, title, block, start FROM sections ORDER BY id")
        .map_err(sql)?;
    let sections: Vec<Section> = q
        .query_map([], |r| {
            let block: Option<String> = r.get(4)?;
            let start: Option<i64> = r.get(5)?;
            Ok(Section {
                id: r.get(0)?,
                parent: r.get::<_, Option<i32>>(1)?.unwrap_or(-1),
                level: r.get(2)?,
                title: r.get(3)?,
                block: block
                    .map(|b| piece_id(&b, start.unwrap_or(0) as usize))
                    .unwrap_or_default(),
            })
        })
        .map_err(sql)?
        .collect::<Result<_, _>>()
        .map_err(sql)?;

    let mut refs: Vec<(i32, Vec<String>)> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut q = db
        .prepare("SELECT c.section, v.vector FROM chunks c LEFT JOIN vectors v ON v.key = c.vector ORDER BY c.id")
        .map_err(sql)?;
    for r in q
        .query_map([], |r| {
            Ok((r.get::<_, i32>(0)?, r.get::<_, Option<Vec<u8>>>(1)?))
        })
        .map_err(sql)?
    {
        let (s, v) = r.map_err(sql)?;
        refs.push((s, Vec::new()));
        if let Some(v) = v {
            rows.push(
                v.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            );
        }
    }
    let mut q = db
        .prepare("SELECT chunk, block, start FROM chunk_blocks ORDER BY chunk, pos")
        .map_err(sql)?;
    for r in q
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })
        .map_err(sql)?
    {
        let (c, b, s) = r.map_err(sql)?;
        refs.get_mut(c as usize)
            .ok_or("chunk_blocks names a missing chunk")?
            .1
            .push(piece_id(&b, s as usize));
    }
    let mut floats = BTreeMap::new();
    let mut q = db
        .prepare("SELECT label, section FROM floats")
        .map_err(sql)?;
    for r in q
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(sql)?
    {
        let (k, v) = r.map_err(sql)?;
        floats.insert(k, v);
    }
    let mut q = db
        .prepare("SELECT type, source, target, label FROM edges")
        .map_err(sql)?;
    let edges: Vec<Edge> = q
        .query_map([], |r| {
            Ok(Edge {
                etype: r.get(0)?,
                from: r.get(1)?,
                to: r.get(2)?,
                label: r.get(3)?,
            })
        })
        .map_err(sql)?
        .collect::<Result<_, _>>()
        .map_err(sql)?;
    let mut edits = Edits::default();
    for (block, start, stop, role, at) in edit_rows(&db)? {
        let role = Role::parse(&role).ok_or_else(|| format!("edits: unknown role {role}"))?;
        edits.0.push(Edit {
            block,
            start: start as usize,
            stop: stop.map(|n| n as usize),
            role,
            at,
        });
    }

    let title = sections
        .first()
        .map(|s| s.title.clone())
        .unwrap_or_default();
    let tree = Tree::assemble(
        title,
        source,
        sections,
        refs,
        floats,
        &edits.pieces(&blocks),
    )?;
    let vectors = if rows.is_empty() {
        Tensor::default()
    } else {
        Tensor::from_rows(rows)?
    };
    if vectors.rows > 0 && vectors.rows != tree.chunks.len() {
        return Err(format!(
            "{}: {} vectors for {} excerpts",
            path.display(),
            vectors.rows,
            tree.chunks.len()
        )
        .into());
    }
    Ok(Package {
        embed_model: embed_model.unwrap_or_default(),
        source_sha256,
        chunking: ParserOptions {
            target_tokens,
            max_tokens,
        },
        blocks,
        pages,
        tree,
        edits,
        vectors,
        edges,
    })
}

/// Reaproveita os vetores de trechos cujo texto não mudou; `None` indica vetor a calcular.
pub fn carry_vectors(old: &Tree, vectors: &Tensor, new: &Tree) -> Vec<Option<Vec<f32>>> {
    let mut by_text: HashMap<String, usize> = HashMap::new();
    if vectors.rows == old.chunks.len() {
        for (i, c) in old.chunks.iter().enumerate() {
            by_text.entry(old.embedding_text(c)).or_insert(i);
        }
    }
    new.chunks
        .iter()
        .map(|c| {
            by_text
                .get(&new.embedding_text(c))
                .map(|&i| vectors.row(i).to_vec())
        })
        .collect()
}

pub struct Hit {
    pub chunk: usize,
    pub score: f32,
}

/// Os k trechos de maior similaridade de cosseno com a consulta (vetores normalizados).
pub fn top_k(
    vectors: &Tensor,
    query: &[f32],
    k: usize,
    keep: impl Fn(usize) -> bool,
) -> Res<Vec<Hit>> {
    if vectors.rows == 0 {
        return Err("search: package has no vectors (compiled with --no-embed?)".into());
    }
    if query.len() != vectors.dim {
        return Err(format!(
            "search: query dimension ({}) != index dimension ({})",
            query.len(),
            vectors.dim
        )
        .into());
    }
    let mut hits: Vec<Hit> = (0..vectors.rows)
        .filter(|&i| keep(i))
        .map(|i| Hit {
            chunk: i,
            score: vectors.row(i).iter().zip(query).map(|(a, b)| a * b).sum(),
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    hits.truncate(k);
    Ok(hits)
}
