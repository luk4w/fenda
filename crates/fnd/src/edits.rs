//! Correções do sumário.
//!
//! O sumário é derivado de `blocos + edições`: o parser classifica os títulos e cada edição fixa o papel (`Role`) de um fragmento de bloco. Remover a edição restaura a classificação automática.
//! Um fragmento pode ser parte de um bloco (título fundido ao texto pelo OCR) ou a união de blocos consecutivos (`Join`, título dividido em duas linhas).

use crate::tree::{Block, BlockType};
use std::collections::HashMap;

/// Papel de uma linha no sumário.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Chapter,
    Section,
    Subsection,
    Heading,
    Text,
    Title,
    Join,
}

impl Role {
    pub const ALL: [Role; 7] = [
        Role::Chapter,
        Role::Section,
        Role::Subsection,
        Role::Heading,
        Role::Text,
        Role::Title,
        Role::Join,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Role::Chapter => "chapter",
            Role::Section => "section",
            Role::Subsection => "subsection",
            Role::Heading => "heading",
            Role::Text => "text",
            Role::Title => "title",
            Role::Join => "join",
        }
    }

    pub fn parse(s: &str) -> Option<Role> {
        Role::ALL.into_iter().find(|r| r.name() == s)
    }

    /// Profundidade no sumário (capítulo = 1); `None` para papéis que não abrem seção.
    pub fn level(self) -> Option<i32> {
        match self {
            Role::Chapter => Some(1),
            Role::Section => Some(2),
            Role::Subsection => Some(3),
            Role::Heading | Role::Text | Role::Title | Role::Join => None,
        }
    }

    pub fn opens_section(self) -> bool {
        matches!(
            self,
            Role::Chapter | Role::Section | Role::Subsection | Role::Heading
        )
    }
}

/// Correção de um fragmento: bloco, intervalo em bytes, papel e data.
#[derive(Debug, Clone)]
pub struct Edit {
    pub block: String,
    pub start: usize,
    pub stop: Option<usize>,
    pub role: Role,
    pub at: String,
}

impl Edit {
    pub fn piece(&self) -> String {
        piece_id(&self.block, self.start)
    }
}

/// Identificador do fragmento: id do bloco, acrescido de `@início` quando o fragmento não começa no byte 0.
pub fn piece_id(block: &str, start: usize) -> String {
    if start == 0 {
        block.to_string()
    } else {
        format!("{block}@{start}")
    }
}

pub fn split_piece(id: &str) -> (&str, usize) {
    match id.rsplit_once('@') {
        Some((b, s)) => match s.parse() {
            Ok(n) => (b, n),
            Err(_) => (id, 0),
        },
        None => (id, 0),
    }
}

/// Conjunto de edições de um livro.
#[derive(Debug, Clone, Default)]
pub struct Edits(pub Vec<Edit>);

impl Edits {
    pub fn get(&self, piece: &str) -> Option<&Edit> {
        let (block, start) = split_piece(piece);
        self.0.iter().find(|e| e.block == block && e.start == start)
    }

    pub fn set(&mut self, e: Edit) {
        self.remove(&e.piece());
        self.0.push(e);
    }

    pub fn remove(&mut self, piece: &str) {
        let (block, start) = split_piece(piece);
        self.0.retain(|e| !(e.block == block && e.start == start));
    }

    /// Intervalos em que as edições dividem um bloco.
    pub fn ranges(&self, b: &Block) -> Vec<(usize, usize)> {
        let len = b.text.len();
        let mut cuts: Vec<usize> = self
            .0
            .iter()
            .filter(|e| e.block == b.id)
            .flat_map(|e| [Some(e.start), e.stop].into_iter().flatten())
            .filter(|&c| c > 0 && c < len && b.text.is_char_boundary(c))
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let mut from = 0;
        for c in cuts.into_iter().chain([len]) {
            out.push((from, c));
            from = c;
        }
        out
    }

    pub fn title(&self) -> Option<&Edit> {
        self.0.iter().find(|e| e.role == Role::Title)
    }

    /// Blocos após a aplicação dos cortes e uniões definidos pelas edições.
    pub fn pieces(&self, blocks: &[Block]) -> Vec<Block> {
        self.join(self.cut(blocks))
    }

    /// Fragmentos unidos a este por `Join`.
    pub fn joined_after(&self, blocks: &[Block], piece: &str) -> Vec<String> {
        let (block, _) = split_piece(piece);
        let Some(page) = blocks.iter().find(|b| b.id == block).map(|b| b.page) else {
            return Vec::new();
        };
        let on_page: Vec<Block> = blocks.iter().filter(|b| b.page == page).cloned().collect();
        let cut = self.cut(&on_page);
        let Some(i) = cut.iter().position(|b| b.id == piece) else {
            return Vec::new();
        };
        cut[i + 1..]
            .iter()
            .take_while(|b| self.get(&b.id).is_some_and(|e| e.role == Role::Join))
            .map(|b| b.id.clone())
            .collect()
    }

    fn join(&self, pieces: Vec<Block>) -> Vec<Block> {
        let joins: std::collections::HashSet<String> = self
            .0
            .iter()
            .filter(|e| e.role == Role::Join)
            .map(Edit::piece)
            .collect();
        if joins.is_empty() {
            return pieces;
        }
        let mut out: Vec<Block> = Vec::with_capacity(pieces.len());
        for b in pieces {
            match out.last_mut() {
                Some(prev) if joins.contains(&b.id) && prev.page == b.page => {
                    prev.text.push('\n');
                    prev.text.push_str(&b.text);
                    if prev.bbox.len() == 4 && b.bbox.len() == 4 {
                        prev.bbox = vec![
                            prev.bbox[0].min(b.bbox[0]),
                            prev.bbox[1].min(b.bbox[1]),
                            prev.bbox[2].max(b.bbox[2]),
                            prev.bbox[3].max(b.bbox[3]),
                        ];
                    }
                }
                _ => out.push(b),
            }
        }
        out
    }

    fn cut(&self, blocks: &[Block]) -> Vec<Block> {
        let cut: std::collections::HashSet<&str> = self
            .0
            .iter()
            .filter(|e| e.start > 0 || e.stop.is_some())
            .map(|e| e.block.as_str())
            .collect();
        let mut out = Vec::with_capacity(blocks.len() + cut.len() * 2);
        for b in blocks {
            if !cut.contains(b.id.as_str()) {
                out.push(b.clone());
                continue;
            }
            for (from, to) in self.ranges(b) {
                let text = b.text[from..to].trim();
                if text.is_empty() {
                    continue;
                }
                out.push(Block {
                    id: piece_id(&b.id, from),
                    text: text.to_string(),
                    ..b.clone()
                });
            }
        }
        out
    }

    /// Aplica os papéis aos fragmentos e retorna a quantidade alterada.
    pub fn apply(&self, blocks: &mut [Block]) -> usize {
        if self.0.is_empty() {
            return 0;
        }
        let pieces: Vec<String> = self.0.iter().map(Edit::piece).collect();
        let index: HashMap<&str, &Edit> = pieces.iter().map(String::as_str).zip(&self.0).collect();
        let mut applied = 0;
        for b in blocks.iter_mut() {
            if !matches!(b.btype, BlockType::Heading | BlockType::Paragraph) {
                continue;
            }
            let Some(e) = index.get(b.id.as_str()).filter(|e| e.role != Role::Join) else {
                continue;
            };
            b.btype = if e.role.opens_section() {
                BlockType::Heading
            } else {
                BlockType::Paragraph
            };
            b.forced_level = e.role.level();
            applied += 1;
        }
        applied
    }
}

/// Normaliza o texto para comparar a citação da IA com o texto reconhecido pelo OCR.
pub fn plain(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

pub fn today() -> String {
    date(std::time::SystemTime::now())
}

pub fn date(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}
