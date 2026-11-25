//! Montagem do sumário a partir dos blocos.
//!
//! Não há regras por livro ou idioma: os candidatos a título vêm do layout e são validados por evidências do próprio documento (sumário impresso, numeração, marcador de capítulo inferido dos títulos). Casos não resolvidos ficam para correção pela IA (`restructure`).
//!
//! A mesma rotina é executada na compilação e a cada edição, o que garante sumários idênticos. Títulos são comparados por byte, não por caractere.

use crate::edits::Edits;
use crate::tree::{
    Block, BlockType, Chunk, Edge, Section, TITLE_LABELS, Tree, estimate_tokens, subtitle,
};
use regex::bytes::Regex;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::OnceLock;

pub fn trim(s: &[u8]) -> &[u8] {
    let mut a = 0;
    let mut b = s.len();
    while a < b && matches!(s[a], b' ' | b'\t' | b'\r') {
        a += 1;
    }
    while b > a && matches!(s[b - 1], b' ' | b'\t' | b'\r') {
        b -= 1;
    }
    &s[a..b]
}

pub fn trim_str(s: &str) -> &str {
    std::str::from_utf8(trim(s.as_bytes())).unwrap_or(s)
}

pub fn fold_lower(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if c == 0xC3 && i + 1 < s.len() {
            let d = s[i + 1];
            out.push(0xC3);
            out.push(if (0x80..=0x9E).contains(&d) && d != 0x97 {
                d + 0x20
            } else {
                d
            });
            i += 2;
        } else {
            out.push(c.to_ascii_lowercase());
            i += 1;
        }
    }
    out
}

#[derive(Debug, Clone, Default)]
pub struct HeadingNum {
    pub ok: bool,
    pub depth: i32,
    pub chapter: i64,
    pub chapter_is_int: bool,
    pub letter: u8,
    pub path: String,
}

/// Extrai a numeração de um título ("1", "4.3.2", "A.1"); exige separador após o número ("6Texto" não é numerado).
pub fn parse_heading_number(s: &str) -> HeadingNum {
    let s = trim(s.as_bytes());
    let mut r = HeadingNum {
        chapter: -1,
        ..Default::default()
    };
    if s.is_empty() {
        return r;
    }
    if s[0].is_ascii_digit() {
        let mut i = 0usize;
        let mut comps = 0i32;
        let mut first: i64 = -1;
        let mut second: i64 = -1;
        let mut used_hyphen = false;
        while i < s.len() && s[i].is_ascii_digit() {
            let mut v: i64 = 0;
            let d0 = i;
            while i < s.len() && s[i].is_ascii_digit() {
                v = v * 10 + (s[i] - b'0') as i64;
                i += 1;
            }
            if i - d0 > 4 {
                return r;
            }
            if comps == 0 {
                first = v;
            } else if comps == 1 {
                second = v;
            }
            comps += 1;
            if i + 1 < s.len() && s[i] == b'.' && s[i + 1].is_ascii_digit() {
                i += 1;
                continue;
            }
            if comps == 1
                && (1..=50).contains(&first)
                && i + 1 < s.len()
                && s[i] == b'-'
                && s[i + 1].is_ascii_digit()
            {
                used_hyphen = true;
                i += 1;
                continue;
            }
            break;
        }
        if used_hyphen && (comps != 2 || !(1..=50).contains(&second)) {
            return r;
        }
        let mut j = i;
        if j < s.len() && (s[j] == b'.' || s[j] == b')') {
            j += 1;
        }
        if j == s.len() || s[j] == b' ' || s[j] == b'\t' {
            r.ok = true;
            r.depth = comps;
            r.chapter = first;
            r.chapter_is_int = true;
            r.path = if used_hyphen {
                format!("{}.{}", first, second)
            } else {
                String::from_utf8_lossy(&s[..i]).into_owned()
            };
        }
        return r;
    }
    if s.len() >= 2 && s[0].is_ascii_uppercase() && s[1] == b'.' {
        let mut i = 1usize;
        let mut comps = 1i32;
        while i + 1 < s.len() && s[i] == b'.' && s[i + 1].is_ascii_digit() {
            i += 1;
            while i < s.len() && s[i].is_ascii_digit() {
                i += 1;
            }
            comps += 1;
        }
        let mut j = i;
        if j < s.len() && (s[j] == b'.' || s[j] == b')') {
            j += 1;
        }
        if j == s.len() || s[j] == b' ' {
            r.ok = true;
            r.depth = comps;
            r.letter = s[0];
            r.path = String::from_utf8_lossy(&s[..i]).into_owned();
        }
    }
    r
}

/// Rebaixa a texto os termos de glossário de margem: títulos sem número que se repetem no texto vizinho e têm pouco conteúdo próprio.
pub fn demote_margin_headings(blocks: &mut [Block]) -> usize {
    let tx: Vec<usize> = (0..blocks.len())
        .filter(|&i| {
            blocks[i].btype == BlockType::Heading || blocks[i].btype == BlockType::Paragraph
        })
        .collect();

    let mut demote: Vec<usize> = Vec::new();
    for k in 0..tx.len() {
        let b = &blocks[tx[k]];
        if b.btype != BlockType::Heading {
            continue;
        }
        let title = trim(b.text.as_bytes()).to_vec();
        if title.len() < 4 || parse_heading_number(&String::from_utf8_lossy(&title)).ok {
            continue;
        }
        let t = fold_lower(&title);

        let mut ctx: Vec<u8> = Vec::new();
        for off in [-1i64, 1, 2] {
            let j = k as i64 + off;
            if j >= 0 && (j as usize) < tx.len() {
                let nb = &blocks[tx[j as usize]];
                if nb.btype == BlockType::Paragraph {
                    ctx.extend_from_slice(&fold_lower(nb.text.as_bytes()));
                }
            }
        }
        let mut own = 0usize;
        let mut j = k + 1;
        while j < tx.len() && blocks[tx[j]].btype == BlockType::Paragraph {
            own += 1;
            j += 1;
        }

        if own <= 3 && contains_bytes(&ctx, &t) {
            demote.push(tx[k]);
        }
    }
    for i in &demote {
        blocks[*i].btype = BlockType::Paragraph;
    }
    demote.len()
}

/// Chave de comparação entre uma linha do sumário impresso e o título correspondente no corpo.
pub fn title_key(s: &str) -> String {
    let mut plain = String::with_capacity(s.len());
    for c in s.chars().flat_map(char::to_lowercase) {
        match c {
            'á' | 'à' | 'â' | 'ã' | 'ä' | 'å' => plain.push('a'),
            'é' | 'è' | 'ê' | 'ë' => plain.push('e'),
            'í' | 'ì' | 'î' | 'ï' => plain.push('i'),
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' => plain.push('o'),
            'ú' | 'ù' | 'û' | 'ü' => plain.push('u'),
            'ç' => plain.push('c'),
            'ñ' => plain.push('n'),
            'ﬁ' => plain.push_str("fi"),
            'ﬂ' => plain.push_str("fl"),
            _ => plain.push(c),
        }
    }
    static WORD: OnceLock<regex::Regex> = OnceLock::new();
    static NUMBER: OnceLock<regex::Regex> = OnceLock::new();
    let word = WORD
        .get_or_init(|| regex::Regex::new(r"^\s*(parte|capitulo|chapter|apendice)\s+").unwrap());
    let number = NUMBER.get_or_init(|| regex::Regex::new(r"^\s*\d+(\.\d+)*\s*[.)\-]?\s*").unwrap());
    let plain = plain.trim_start_matches(|c: char| !c.is_alphanumeric());
    let rest = word.replace(plain, " ");
    let rest = number.replace(&rest, " ");
    rest.chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Confirmation {
    pub confirmed: usize,
    pub unconfirmed: usize,
}

/// Confirma ou rebaixa candidatos a título com base no sumário impresso, quando existir.
pub fn confirm_by_contents(blocks: &mut [Block]) -> Confirmation {
    static PAGE_NUMBER: OnceLock<regex::Regex> = OnceLock::new();
    let page_number =
        PAGE_NUMBER.get_or_init(|| regex::Regex::new(r"[\s.·…_]*\d{1,4}\s*$").unwrap());

    let mut listed: HashSet<String> = HashSet::new();
    let mut contents_pages: HashSet<i32> = HashSet::new();
    for b in blocks.iter().filter(|b| b.label == "content") {
        contents_pages.insert(b.page);
        for line in b.text.split(['\n', '•']) {
            let k = title_key(&page_number.replace(line, ""));
            if k.len() >= 4 {
                listed.insert(k);
            }
        }
    }
    let mut out = Confirmation::default();
    if listed.is_empty() {
        return out;
    }
    let long: Vec<&String> = listed.iter().filter(|t| t.len() >= 12).collect();
    let is_listed = |k: &str| {
        listed.contains(k)
            || (k.len() >= 12
                && long
                    .iter()
                    .any(|t| t.starts_with(k) || k.starts_with(t.as_str())))
    };

    for i in 0..blocks.len() {
        let b = &blocks[i];
        if !TITLE_LABELS.contains(&b.label.as_str()) || contents_pages.contains(&b.page) {
            continue;
        }
        if is_listed(&title_key(&b.text)) {
            blocks[i].btype = BlockType::Heading;
            out.confirmed += 1;
        } else if b.label != "doc_title"
            && !parse_heading_number(&b.text).ok
            && palavra_e_numero(&b.text).is_none()
        {
            if b.btype == BlockType::Heading {
                out.unconfirmed += 1;
            }
            blocks[i].btype = BlockType::Paragraph;
        }
    }
    out
}

pub fn contains_bytes(agulheiro: &[u8], agulha: &[u8]) -> bool {
    if agulha.is_empty() {
        return true;
    }
    if agulha.len() > agulheiro.len() {
        return false;
    }
    agulheiro.windows(agulha.len()).any(|w| w == agulha)
}

#[derive(Debug, Clone, Default)]
pub struct Marcador {
    pub palavra: String,
    pub onde: BTreeMap<usize, i64>,
}

pub fn palavra_e_numero(s: &str) -> Option<(String, i64)> {
    let s = trim(s.as_bytes());
    let mut i = 0usize;
    while i < s.len() && !s[i].is_ascii_whitespace() {
        i += 1;
    }
    if i == 0 || i > 24 || s[0].is_ascii_digit() {
        return None;
    }
    let w = i;
    while i < s.len() && (s[i] == b' ' || s[i] == b'\t') {
        i += 1;
    }
    let d0 = i;
    let mut v: i64 = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        v = v * 10 + (s[i] - b'0') as i64;
        i += 1;
    }
    if i == d0 || i - d0 > 3 {
        return None;
    }
    if i < s.len() && !s[i].is_ascii_whitespace() {
        return None;
    }
    Some((
        String::from_utf8_lossy(&fold_lower(&s[..w])).into_owned(),
        v,
    ))
}

/// Infere a palavra de capítulo do livro ("Chapter", "Capítulo"…) a partir dos próprios títulos.
pub fn learn_chapter_marker(titles: &[String]) -> Marcador {
    let mut tem1: BTreeSet<i64> = BTreeSet::new();
    let mut pedidos: BTreeSet<i64> = BTreeSet::new();
    for t in titles {
        let h = parse_heading_number(t);
        if !h.ok || !h.chapter_is_int || h.chapter < 1 || h.chapter > 200 {
            continue;
        }
        if h.depth == 1 {
            tem1.insert(h.chapter);
        } else {
            pedidos.insert(h.chapter);
        }
    }
    let orfaos: BTreeSet<i64> = pedidos.difference(&tem1).copied().collect();
    let mut m = Marcador::default();
    if orfaos.is_empty() {
        return m;
    }

    let mut cand: BTreeMap<String, Vec<(usize, i64)>> = BTreeMap::new();
    for (i, t) in titles.iter().enumerate() {
        if let Some((p, v)) = palavra_e_numero(t) {
            if (1..=200).contains(&v) {
                cand.entry(p).or_default().push((i, v));
            }
        }
    }
    let mut melhor = 0usize;
    let mut ambigua = false;
    for (palavra, oc) in &cand {
        if oc.len() < 3 {
            continue;
        }
        let mut cresce = true;
        let mut na_lacuna = 0usize;
        for j in 0..oc.len() {
            if j > 0 && oc[j].1 <= oc[j - 1].1 {
                cresce = false;
            }
            if orfaos.contains(&oc[j].1) {
                na_lacuna += 1;
            }
        }
        if !cresce || na_lacuna * 2 < oc.len() {
            continue;
        }
        if oc.len() > melhor {
            melhor = oc.len();
            ambigua = false;
            m.palavra = palavra.clone();
            m.onde.clear();
            for (idx, v) in oc {
                m.onde.insert(*idx, *v);
            }
        } else if oc.len() == melhor {
            ambigua = true;
        }
    }
    if ambigua {
        return Marcador::default();
    }
    m
}

#[derive(Debug, Clone, Default)]
pub struct Hierarquia {
    pub depth: Vec<i32>,
    pub parent: Vec<i32>,
    pub declarado: i32,
    pub orfao: i32,
}

/// Numeração de cada título ("4.2", "A.1", "Capítulo 7"), aceita apenas quando coerente com a sequência do livro ("E. coli" não é o apêndice E).
pub fn monotonic_numbers(titles: &[String]) -> Vec<HeadingNum> {
    let n = titles.len();
    let mut num: Vec<HeadingNum> = vec![HeadingNum::default(); n];
    let mut app_cur: u8 = 0;
    let marc = learn_chapter_marker(titles);
    let mut int_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        let mut h = parse_heading_number(&titles[i]);
        if !h.ok {
            if let Some(cap) = marc.onde.get(&i) {
                h.ok = true;
                h.depth = 1;
                h.chapter = *cap;
                h.chapter_is_int = true;
                h.path = cap.to_string();
            }
        }
        if h.ok && h.chapter_is_int {
            if h.chapter >= 1 && h.chapter <= 200 {
                int_idx.push(i);
            } else {
                h.ok = false;
            }
        } else if h.ok && h.letter != 0 {
            let expect = if app_cur != 0 { app_cur + 1 } else { b'A' };
            if h.letter == app_cur || h.letter == expect {
                app_cur = std::cmp::max(app_cur, h.letter);
            } else {
                h.ok = false;
            }
        }
        num[i] = h;
    }

    if !int_idx.is_empty() {
        let m = int_idx.len();
        let mut dp = vec![1usize; m];
        let mut prv = vec![usize::MAX; m];
        for a in 0..m {
            let ia = int_idx[a];
            let cha = num[ia].chapter;
            let da = num[ia].depth;
            for b in 0..a {
                let ib = int_idx[b];
                let chb = num[ib].chapter;
                let ok_step = if da == 1 {
                    cha > chb && cha <= chb + 5
                } else {
                    cha >= chb && cha <= chb + 5
                };
                if ok_step && dp[b] + 1 > dp[a] {
                    dp[a] = dp[b] + 1;
                    prv[a] = b;
                }
            }
        }
        let mut best_end = 0usize;
        for a in 1..m {
            if dp[a] > dp[best_end] {
                best_end = a;
            }
        }
        let mut keep = vec![false; n];
        let mut cur = best_end;
        while cur != usize::MAX {
            keep[int_idx[cur]] = true;
            cur = prv[cur];
        }
        for &idx in &int_idx {
            if !keep[idx] {
                num[idx].ok = false;
            }
        }
    }
    num
}

/// Nível de cada título, na ordem de leitura.
pub fn resolve_hierarchy(titles: &[String], labels: &[String]) -> Hierarquia {
    let n = titles.len();
    let mut num = monotonic_numbers(titles);

    {
        let mut tem1: BTreeSet<i64> = BTreeSet::new();
        let mut pedidos: BTreeSet<i64> = BTreeSet::new();
        for h in num.iter() {
            if !h.ok || !h.chapter_is_int {
                continue;
            }
            if h.depth == 1 {
                tem1.insert(h.chapter);
            } else {
                pedidos.insert(h.chapter);
            }
        }
        for cap in pedidos.iter().copied() {
            if tem1.contains(&cap) {
                continue;
            }
            let mut j = n;
            for i in 0..n {
                if num[i].ok && num[i].chapter_is_int && num[i].chapter == cap && num[i].depth >= 2
                {
                    j = i;
                    break;
                }
            }
            if j == n {
                continue;
            }
            let mut piso = 0usize;
            for i in 0..j {
                if num[i].ok && num[i].depth == 1 {
                    piso = i + 1;
                }
            }
            let mut i = j;
            while i > piso {
                i -= 1;
                if num[i].ok || labels[i] != "doc_title" {
                    continue;
                }
                num[i].ok = true;
                num[i].depth = 1;
                num[i].chapter = cap;
                num[i].chapter_is_int = true;
                num[i].path = cap.to_string();
                break;
            }
        }
    }

    let keys: Vec<Vec<u8>> = titles
        .iter()
        .map(|t| fold_lower(trim(t.as_bytes())))
        .collect();
    let mut freq: BTreeMap<&[u8], i32> = BTreeMap::new();
    for k in &keys {
        *freq.entry(k.as_slice()).or_insert(0) += 1;
    }

    let mut depth = vec![1i32; n];
    let mut seen = false;
    let mut chapter_claimed = false;
    let mut last_ch: i64 = 0;
    let mut cur_section: i32 = -1;
    for i in 0..n {
        if num[i].ok {
            depth[i] = num[i].depth;
            seen = true;
            chapter_claimed = false;
            if num[i].chapter_is_int {
                last_ch = std::cmp::max(last_ch, num[i].chapter);
            }
            cur_section = depth[i];
        } else {
            let mut jn = i + 1;
            while jn < n && !num[jn].ok {
                jn += 1;
            }
            let nxt = if jn < n { Some(&num[jn]) } else { None };
            let is_recurring = freq[keys[i].as_slice()] >= 2;
            let at_chapter_end = nxt.is_none_or(|x| x.chapter_is_int && x.chapter > last_ch);
            if !seen {
                depth[i] = 1;
            } else if at_chapter_end && is_recurring {
                depth[i] = 2;
                cur_section = 2;
            } else if !chapter_claimed
                && at_chapter_end
                && !is_recurring
                && nxt.is_none_or(|x| x.depth >= 2)
            {
                depth[i] = 1;
                cur_section = 1;
                if nxt.is_some() {
                    chapter_claimed = true;
                }
            } else if cur_section >= 1 {
                depth[i] = cur_section + 1;
            } else {
                depth[i] = 1;
            }
        }
    }

    let mut por_caminho: BTreeMap<&str, usize> = BTreeMap::new();
    for i in 0..n {
        if num[i].ok && !num[i].path.is_empty() {
            por_caminho.entry(num[i].path.as_str()).or_insert(i);
        }
    }

    let mut parent = vec![-1i32; n];
    let mut declarado = 0i32;
    let mut orfao = 0i32;
    for i in 0..n {
        if !num[i].ok || num[i].depth < 2 || num[i].path.is_empty() {
            continue;
        }
        let corte = match num[i].path.rfind('.') {
            Some(c) => c,
            None => continue,
        };
        match por_caminho.get(&num[i].path[..corte]) {
            Some(&p) if p < i => {
                parent[i] = p as i32;
                declarado += 1;
            }
            _ => orfao += 1,
        }
    }

    Hierarquia {
        depth,
        parent,
        declarado,
        orfao,
    }
}

#[derive(Debug, Clone, Default)]
pub struct Conferencia {
    pub declarado: i32,
    pub violado: i32,
    pub orfao: i32,
    pub exemplos: Vec<String>,
}

/// Verifica o sumário contra a numeração do livro (toda seção "4.2" sob o capítulo "4").
pub fn check_numbering(tree: &Tree) -> Conferencia {
    let mut c = Conferencia::default();
    let titles: Vec<String> = tree
        .sections
        .iter()
        .filter(|s| s.id != 0)
        .map(|s| s.title.clone())
        .collect();
    let num = monotonic_numbers(&titles);
    let mut por_caminho: BTreeMap<String, i32> = BTreeMap::new();
    let mut idx = 0usize;
    for s in &tree.sections {
        if s.id == 0 {
            continue;
        }
        let h = &num[idx];
        idx += 1;
        if h.ok && !h.path.is_empty() {
            por_caminho.entry(h.path.clone()).or_insert(s.id);
        }
    }
    let mut idx = 0usize;
    for s in &tree.sections {
        if s.id == 0 {
            continue;
        }
        let h = &num[idx];
        idx += 1;
        if !h.ok || h.depth < 2 || h.path.is_empty() {
            continue;
        }
        let corte = match h.path.rfind('.') {
            Some(x) => x,
            None => continue,
        };
        let pai = match por_caminho.get(&h.path[..corte]) {
            Some(&p) if p <= s.id => p,
            _ => continue,
        };
        if s.parent == pai {
            c.declarado += 1;
        } else {
            c.violado += 1;
            if c.exemplos.len() < 5 {
                let sob = if s.parent >= 0 {
                    tree.sections[s.parent as usize].title.clone()
                } else {
                    "(root)".to_string()
                };
                c.exemplos.push(format!(
                    "{} belongs under \"{}\" but would sit under \"{}\"",
                    h.path, tree.sections[pai as usize].title, sob
                ));
            }
        }
    }
    c
}

pub fn leading_number(s: &str) -> String {
    let s = trim(s.as_bytes());
    let mut i = 0usize;
    if s.is_empty() || !s[0].is_ascii_digit() {
        return String::new();
    }
    while i < s.len()
        && (s[i].is_ascii_digit() || (s[i] == b'.' && i + 1 < s.len() && s[i + 1].is_ascii_digit()))
    {
        i += 1;
    }
    String::from_utf8_lossy(&s[..i]).into_owned()
}

fn re_wn() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?-u)([A-Z][^ \t\r\n0-9][^ \t\r\n]{0,28})[ \t]+([0-9]+(?:\.[0-9]+)*)").unwrap()
    })
}

fn re_tagn() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?-u)(?:\\tag\{|\()([0-9]+(?:\.[0-9]+)*)").unwrap())
}

fn grupo(m: &regex::bytes::Captures, i: usize) -> String {
    String::from_utf8_lossy(m.get(i).map(|x| x.as_bytes()).unwrap_or(b"")).into_owned()
}

/// Referências cruzadas sem lista de palavras: o termo de seção é o que precede números correspondentes a seções, e não a figuras ou equações.
pub fn extract_edges(tree: &Tree, blocks: &[Block]) -> Vec<Edge> {
    let mut num2id: BTreeMap<String, i32> = BTreeMap::new();
    for s in &tree.sections {
        let n = leading_number(&s.title);
        if !n.is_empty() {
            num2id.entry(n).or_insert(s.id);
        }
    }

    let mut fnums: BTreeSet<String> = BTreeSet::new();
    let mut enums_: BTreeSet<String> = BTreeSet::new();
    let mut fwords: BTreeSet<String> = BTreeSet::new();
    for b in blocks {
        if !b.caption.is_empty() {
            if let Some(m) = re_wn().captures(b.caption.as_bytes()) {
                fwords.insert(grupo(&m, 1).to_ascii_lowercase());
                fnums.insert(grupo(&m, 2));
            }
        }
        if !b.tag.is_empty() {
            enums_.insert(b.tag.clone());
        }
        if b.btype == BlockType::Math {
            for m in re_tagn().captures_iter(b.text.as_bytes()) {
                enums_.insert(grupo(&m, 1));
            }
        }
    }

    let is_float_word = |w: &str| -> bool {
        let w = w.to_ascii_lowercase();
        fwords.iter().any(|s| {
            w == *s
                || (w.len() > s.len() && w.starts_with(s.as_str()) && w.len() - s.len() <= 2)
                || (s.len() > w.len() && s.starts_with(w.as_str()) && s.len() - w.len() <= 2)
        })
    };

    let mut edges: Vec<Edge> = Vec::new();
    let mut seen_float: HashSet<(i32, i32)> = HashSet::new();
    let mut seen_cross: HashSet<(i32, i32)> = HashSet::new();
    let mut hits: BTreeMap<String, Vec<(i32, String)>> = BTreeMap::new();
    for c in &tree.chunks {
        for m in re_wn().captures_iter(c.text.as_bytes()) {
            let w = grupo(&m, 1);
            let n = grupo(&m, 2);
            if is_float_word(&w) {
                if let Some(&dest) = tree.floats.get(&n) {
                    if dest != c.section && seen_float.insert((c.section, dest)) {
                        edges.push(Edge {
                            etype: "reference".into(),
                            from: c.section,
                            to: dest,
                            label: format!("{} {}", w, n),
                        });
                    }
                }
                continue;
            }
            hits.entry(w).or_default().push((c.section, n));
        }
    }
    for (w, occ) in &hits {
        let section_word = occ.iter().any(|(_, n)| {
            n.contains('.') && num2id.contains_key(n) && !enums_.contains(n) && !fnums.contains(n)
        });
        if !section_word {
            continue;
        }
        for (sec, n) in occ {
            let dest = match num2id.get(n) {
                Some(&d) if d != *sec => d,
                _ => continue,
            };
            if !seen_cross.insert((*sec, dest)) {
                continue;
            }
            edges.push(Edge {
                etype: "section_reference".into(),
                from: *sec,
                to: dest,
                label: format!("{} {}", w, n),
            });
        }
    }
    edges
}

#[derive(Debug, Clone, Copy)]
pub struct ParserOptions {
    pub target_tokens: i32,
    pub max_tokens: i32,
}

impl Default for ParserOptions {
    fn default() -> Self {
        ParserOptions {
            target_tokens: 300,
            max_tokens: 500,
        }
    }
}

/// Parser configurado com as edições do livro.
pub struct Parser {
    opt: ParserOptions,
    blocks: Vec<Block>,
    hier: Hierarquia,
    min_level: i32,
    first_numbered_h: usize,
    edits: Edits,
    decisions: HeadingDecisions,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HeadingDecisions {
    pub candidates: usize,
    pub dropped: usize,
    pub corrected: usize,
    pub contents: Confirmation,
}

impl Parser {
    pub fn new(opt: ParserOptions) -> Self {
        Parser {
            opt,
            blocks: Vec::new(),
            hier: Hierarquia::default(),
            min_level: 7,
            first_numbered_h: usize::MAX,
            edits: Edits::default(),
            decisions: HeadingDecisions::default(),
        }
    }

    pub fn set_edits(&mut self, edits: Edits) {
        self.edits = edits;
    }

    pub fn decisions(&self) -> HeadingDecisions {
        self.decisions
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn hierarquia(&self) -> &Hierarquia {
        &self.hier
    }

    pub fn parse(&mut self, blocks: &[Block], source_name: &str) -> Tree {
        let mut tree = Tree {
            source: source_name.to_string(),
            ..Default::default()
        };
        tree.sections.push(Section {
            id: 0,
            parent: -1,
            level: 0,
            title: source_name.to_string(),
            block: String::new(),
        });

        self.blocks = blocks
            .iter()
            .map(|b| Block {
                btype: BlockType::of_label(&b.label),
                section: 0,
                forced_level: None,
                ..b.clone()
            })
            .collect();

        let mut decisions = HeadingDecisions {
            candidates: self
                .blocks
                .iter()
                .filter(|b| b.btype == BlockType::Heading)
                .count(),
            ..Default::default()
        };
        decisions.dropped = demote_margin_headings(&mut self.blocks);
        decisions.contents = confirm_by_contents(&mut self.blocks);
        decisions.corrected = self.edits.apply(&mut self.blocks);
        self.decisions = decisions;
        self.assign_sections(&mut tree);

        for b in &self.blocks {
            if b.caption.is_empty() {
                continue;
            }
            if let Some(m) = re_wn().captures(b.caption.as_bytes()) {
                tree.floats.entry(grupo(&m, 2)).or_insert(b.section);
            }
        }

        self.chunk(&mut tree);

        let first_hi = tree
            .sections
            .iter()
            .find(|s| s.id != 0 && s.level == self.min_level)
            .cloned();
        let edited = self.edits.title().and_then(|e| {
            let id = e.piece();
            self.blocks
                .iter()
                .find(|b| b.id == id)
                .map(Self::heading_title)
        });
        if let Some(t) = edited {
            tree.title = t;
        } else if let Some(fh) = first_hi {
            let pretextual = self.first_numbered_h != usize::MAX
                && ((fh.id - 1) as usize) < self.first_numbered_h;
            if !pretextual {
                tree.title = fh.title.clone();
            } else {
                let mut cnt: BTreeMap<&str, i32> = BTreeMap::new();
                let mut order: Vec<&str> = Vec::new();
                for s in &tree.sections {
                    if s.id == 0 {
                        continue;
                    }
                    let pt = self.first_numbered_h != usize::MAX
                        && ((s.id - 1) as usize) < self.first_numbered_h;
                    if !pt {
                        break;
                    }
                    if s.level != self.min_level {
                        continue;
                    }
                    if !cnt.contains_key(s.title.as_str()) {
                        order.push(s.title.as_str());
                    }
                    *cnt.entry(s.title.as_str()).or_insert(0) += 1;
                }
                let mut escolhido = String::new();
                for tt in order {
                    if escolhido.is_empty() || cnt[tt] > cnt[escolhido.as_str()] {
                        escolhido = tt.to_string();
                    }
                }
                tree.title = escolhido;
            }
        }
        if tree.title.is_empty() {
            tree.title = source_name.to_string();
        }
        tree.sections[0].title = tree.title.clone();
        tree
    }

    fn heading_title(b: &Block) -> String {
        let t = trim(b.text.as_bytes());
        let corte = t.iter().position(|&c| c != b'#').unwrap_or(t.len());
        String::from_utf8_lossy(trim(&t[corte..]))
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn assign_sections(&mut self, tree: &mut Tree) {
        let mut titles: Vec<String> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        for b in &self.blocks {
            if b.btype == BlockType::Heading {
                titles.push(Self::heading_title(b));
                labels.push(b.label.clone());
            }
        }

        self.hier = resolve_hierarchy(&titles, &labels);
        for (i, t) in titles.iter().enumerate() {
            if parse_heading_number(t).ok {
                self.first_numbered_h = i;
                break;
            }
        }

        let mut stack: Vec<i32> = vec![0];
        let mut id_do_heading: Vec<i32> = vec![-1; titles.len()];
        let mut h = 0usize;
        for bi in 0..self.blocks.len() {
            if self.blocks[bi].btype == BlockType::Heading {
                let hi = h;
                let mut level = self.hier.depth[h];
                let title = titles[h].clone();
                h += 1;

                let mut pai: i32 = -1;
                if self.hier.parent[hi] >= 0 {
                    pai = id_do_heading[self.hier.parent[hi] as usize];
                }
                if let Some(forced) = self.blocks[bi].forced_level {
                    level = forced;
                    pai = -1;
                }
                if pai >= 0 {
                    level = tree.sections[pai as usize].level + 1;
                } else {
                    while stack.len() > 1
                        && tree.sections[*stack.last().unwrap() as usize].level >= level
                    {
                        stack.pop();
                    }
                    pai = *stack.last().unwrap();
                }
                self.min_level = std::cmp::min(self.min_level, level);

                let s = Section {
                    id: tree.sections.len() as i32,
                    parent: pai,
                    level,
                    title,
                    block: self.blocks[bi].id.clone(),
                };
                let sid = s.id;
                tree.sections.push(s);
                id_do_heading[hi] = sid;
                let mut cadeia: Vec<i32> = Vec::new();
                let mut k = sid;
                while k >= 0 {
                    cadeia.push(k);
                    k = tree.sections[k as usize].parent;
                }
                cadeia.reverse();
                stack = cadeia;
                self.blocks[bi].section = sid;
            } else {
                self.blocks[bi].section = *stack.last().unwrap();
            }
        }
    }

    fn chunk(&self, tree: &mut Tree) {
        let mut cur = Chunk::default();

        macro_rules! flush {
            () => {
                if !cur.blocks.is_empty() {
                    cur.id = tree.chunks.len() as i32;
                    tree.chunks.push(std::mem::take(&mut cur));
                }
            };
        }

        for b in &self.blocks {
            if b.btype == BlockType::Heading {
                flush!();
                continue;
            }
            if b.btype == BlockType::Image {
                continue;
            }

            if !cur.blocks.is_empty() && cur.section != b.section {
                flush!();
            }
            let is_subtitle =
                b.btype == BlockType::Paragraph && TITLE_LABELS.contains(&b.label.as_str());
            if is_subtitle {
                flush!();
            }

            let btok = estimate_tokens(&b.text);
            let ctok = estimate_tokens(&cur.text);
            if !cur.blocks.is_empty()
                && (ctok + btok > self.opt.max_tokens || ctok >= self.opt.target_tokens)
            {
                flush!();
            }

            if cur.blocks.is_empty() {
                cur.section = b.section;
                if is_subtitle {
                    cur.subtitle = subtitle(&b.text);
                }
            }
            cur.push(b);
        }
        flush!();
    }
}

pub struct Built {
    pub tree: Tree,
    pub edges: Vec<Edge>,
    pub decisions: HeadingDecisions,
    pub gate: Conferencia,
}

/// Blocos + edições -> sumário, trechos e referências cruzadas. Executado na compilação e a cada edição pelo MCP.
pub fn build(blocks: &[Block], edits: &Edits, source: &str, opt: ParserOptions) -> Built {
    let mut p = Parser::new(opt);
    p.set_edits(edits.clone());
    let tree = p.parse(&edits.pieces(blocks), source);
    let mut gate = check_numbering(&tree);
    gate.declarado = p.hierarquia().declarado;
    gate.orfao = p.hierarquia().orfao;
    let edges = extract_edges(&tree, p.blocks());
    Built {
        tree,
        edges,
        decisions: p.decisions(),
        gate,
    }
}
