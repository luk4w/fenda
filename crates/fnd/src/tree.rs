//! Blocos e sumário.
//!
//! Um bloco é o conteúdo reconhecido em uma região da página (texto, caixa delimitadora, classe do layout) e é imutável. O sumário (seções e trechos) apenas referencia blocos; corrigi-lo não altera o texto.

use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// Classes do layout que podem abrir uma seção.
pub const TITLE_LABELS: [&str; 2] = ["doc_title", "paragraph_title"];

/// Entrada do sumário; a seção 0 é a raiz e contém o nome do livro.
#[derive(Debug, Clone, Serialize)]
pub struct Section {
    pub id: i32,
    pub parent: i32,
    pub level: i32,
    pub title: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub block: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockType {
    #[default]
    Paragraph,
    Heading,
    Code,
    Math,
    Table,
    Image,
}

impl BlockType {
    pub fn as_str(self) -> &'static str {
        match self {
            BlockType::Paragraph => "paragraph",
            BlockType::Heading => "heading",
            BlockType::Code => "code",
            BlockType::Math => "math",
            BlockType::Table => "table",
            BlockType::Image => "image",
        }
    }

    pub fn of_label(label: &str) -> BlockType {
        match label {
            "image" | "chart" => BlockType::Image,
            "table" => BlockType::Table,
            "algorithm" => BlockType::Code,
            "display_formula" | "inline_formula" => BlockType::Math,
            l if TITLE_LABELS.contains(&l) => BlockType::Heading,
            _ => BlockType::Paragraph,
        }
    }
}

/// Bloco reconhecido: id fixo (`p570_3` = página 570, bloco 3), texto, caixa e classe.
#[derive(Debug, Clone, Default)]
pub struct Block {
    pub id: String,
    pub page: i32,
    pub label: String,
    pub text: String,
    pub caption: String,
    pub tag: String,
    pub bbox: Vec<i32>,
    pub score: f64,
    pub btype: BlockType,
    pub section: i32,
    pub forced_level: Option<i32>,
}

/// Trecho de uma seção, unidade de indexação da busca, com os blocos que o compõem.
#[derive(Debug, Clone, Default)]
pub struct Chunk {
    pub id: i32,
    pub section: i32,
    pub blocks: Vec<String>,
    pub text: String,
    pub tokens_est: i32,
    pub block_types: Vec<String>,
    pub layout: Vec<String>,
    pub page_first: i32,
    pub page_last: i32,
    pub subtitle: String,
    pub marks: Vec<(usize, i32)>,
}

const INDEX_CLASSES: [&str; 2] = ["content", "reference_content"];

/// Estimativa aproximada de tokens, usada apenas para dimensionar respostas.
pub fn estimate_tokens(text: &str) -> i32 {
    std::cmp::max(1, (text.len() / 3) as i32)
}

impl Chunk {
    /// Indica se o trecho é indexado (sumário impresso e bibliografia são excluídos).
    pub fn is_body(&self) -> bool {
        self.layout.is_empty()
            || !self
                .layout
                .iter()
                .all(|l| INDEX_CLASSES.contains(&l.as_str()))
    }

    pub fn push(&mut self, b: &Block) {
        if !self.text.is_empty() {
            self.text.push_str("\n\n");
        }
        self.marks.push((self.text.len(), b.page));
        self.text.push_str(&b.text);
        self.blocks.push(b.id.clone());
        self.block_types.push(b.btype.as_str().to_string());
        self.layout.push(b.label.clone());
        if b.page > 0 {
            if self.page_first == 0 || b.page < self.page_first {
                self.page_first = b.page;
            }
            self.page_last = self.page_last.max(b.page);
        }
        self.tokens_est = estimate_tokens(&self.text);
    }

    pub fn paged_text(&self, page: &mut i32) -> String {
        let mut out = String::with_capacity(self.text.len() + 16 * self.marks.len());
        let mut last = 0;
        for &(start, p) in &self.marks {
            out.push_str(&self.text[last..start]);
            if p > 0 && p != *page {
                out.push_str(&format!("[p. {p}]\n"));
                *page = p;
            }
            last = start;
        }
        out.push_str(&self.text[last..]);
        out
    }
}

/// Referência cruzada entre seções ("ver Seção 4.2").
#[derive(Debug, Clone)]
pub struct Edge {
    pub etype: String,
    pub from: i32,
    pub to: i32,
    pub label: String,
}

/// Sumário montado.
#[derive(Debug, Clone, Default)]
pub struct Tree {
    pub title: String,
    pub source: String,
    pub sections: Vec<Section>,
    pub chunks: Vec<Chunk>,
    pub floats: BTreeMap<String, i32>,
}

#[derive(Serialize)]
struct StoredChunk {
    section: i32,
    blocks: Vec<String>,
}

#[derive(Serialize)]
struct StoredTree {
    title: String,
    source: String,
    sections: Vec<Section>,
    chunks: Vec<StoredChunk>,
    floats: BTreeMap<String, i32>,
}

impl Tree {
    pub fn section_key(&self, s: i32) -> &str {
        &self.sections[s as usize].block
    }

    pub fn section_by_key(&self, key: &str) -> Option<i32> {
        self.sections
            .iter()
            .skip(1)
            .find(|s| s.block == key)
            .map(|s| s.id)
    }

    /// Caminho "Capítulo > Seção > Subseção" de uma seção.
    pub fn breadcrumb(&self, section_id: i32) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut id = section_id;
        while id > 0 {
            parts.push(&self.sections[id as usize].title);
            id = self.sections[id as usize].parent;
        }
        if parts.is_empty() {
            return self.title.clone();
        }
        parts.reverse();
        parts.join(" > ")
    }

    pub fn location(&self, c: &Chunk) -> String {
        let crumb = self.breadcrumb(c.section);
        if c.subtitle.is_empty() {
            crumb
        } else {
            format!("{crumb} > {}", c.subtitle)
        }
    }

    /// Texto submetido ao modelo de embedding: caminho no sumário seguido do conteúdo.
    pub fn embedding_text(&self, c: &Chunk) -> String {
        format!("{} > {}\n{}", self.title, self.location(c), c.text)
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(&StoredTree {
            title: self.title.clone(),
            source: self.source.clone(),
            sections: self.sections.clone(),
            chunks: self
                .chunks
                .iter()
                .map(|c| StoredChunk {
                    section: c.section,
                    blocks: c.blocks.clone(),
                })
                .collect(),
            floats: self.floats.clone(),
        })
    }

    /// Monta o sumário a partir das referências; o texto é obtido dos blocos.
    pub fn assemble(
        title: String,
        source: String,
        sections: Vec<Section>,
        refs: Vec<(i32, Vec<String>)>,
        floats: BTreeMap<String, i32>,
        blocks: &[Block],
    ) -> Result<Tree, String> {
        let by_id: HashMap<&str, &Block> = blocks.iter().map(|b| (b.id.as_str(), b)).collect();
        let mut chunks = Vec::with_capacity(refs.len());
        for (i, (section, ids)) in refs.into_iter().enumerate() {
            let mut c = Chunk {
                id: i as i32,
                section,
                ..Default::default()
            };
            for id in &ids {
                let b = by_id
                    .get(id.as_str())
                    .ok_or_else(|| format!("the tree names block {id}, which does not exist"))?;
                let mut b = (*b).clone();
                b.btype = match BlockType::of_label(&b.label) {
                    BlockType::Heading => BlockType::Paragraph,
                    t => t,
                };
                if c.blocks.is_empty() && TITLE_LABELS.contains(&b.label.as_str()) {
                    c.subtitle = subtitle(&b.text);
                }
                c.push(&b);
            }
            chunks.push(c);
        }
        Ok(Tree {
            title,
            source,
            sections,
            chunks,
            floats,
        })
    }
}

pub fn subtitle(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
