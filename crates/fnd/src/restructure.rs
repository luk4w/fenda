//! Operação `restructure`: a IA define o papel de uma linha (capítulo, seção, título, texto…) e o sumário é remontado imediatamente.
//!
//! A linha é localizada pela página e pelo texto citado, pois a IA não tem acesso aos ids de bloco. Apenas o sumário é alterado; o texto do livro permanece intacto.

use crate::edits::{self, Edit, Role};
use crate::package::{self, Package, Tensor};
use crate::parser;
use crate::tree::{Block, BlockType, TITLE_LABELS, Tree};

/// Alteração solicitada para uma linha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    Set(Role),
    Auto,
}

impl Change {
    pub const NAMES: [&'static str; 7] = [
        "chapter",
        "section",
        "subsection",
        "heading",
        "text",
        "title",
        "auto",
    ];

    pub fn parse(s: &str) -> Option<Change> {
        match s {
            "auto" => Some(Change::Auto),
            "join" => None,
            _ => Role::parse(s).map(Change::Set),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Change::Set(r) => r.name(),
            Change::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Opens(i32),
    Inside(i32),
    Nowhere,
}

/// Posição atual de uma linha no sumário.
pub fn place(tree: &Tree, blocks: &[String]) -> Place {
    if let Some(s) = tree.sections.iter().find(|s| blocks.contains(&s.block)) {
        return Place::Opens(s.id);
    }
    match tree
        .chunks
        .iter()
        .find(|c| c.blocks.iter().any(|b| blocks.contains(b)))
    {
        Some(c) => Place::Inside(c.section),
        None => Place::Nowhere,
    }
}

pub struct Outcome {
    pub package: Package,
    pub blocks: Vec<String>,
    pub sections_before: usize,
    pub reembedded: usize,
    pub unchanged: bool,
    pub cut: bool,
}

fn one_line(s: &str, max: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &flat[..i]),
        None => flat,
    }
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Linha localizada: fragmento inteiro, parte de um bloco ou sequência de linhas a unir.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Pieces(Vec<String>),
    Span {
        block: String,
        start: usize,
        stop: usize,
    },
    Join(Vec<String>),
}

fn locate(text: &str, from: usize, to: usize, wanted: &str) -> Option<Result<(usize, usize), ()>> {
    let mut chars: Vec<(char, usize, usize)> = Vec::new();
    for (i, c) in text[from..to].char_indices() {
        if c.is_alphanumeric() {
            for l in c.to_lowercase() {
                chars.push((l, from + i, from + i + c.len_utf8()));
            }
        }
    }
    let w: Vec<char> = wanted.chars().collect();
    if w.is_empty() || w.len() > chars.len() {
        return None;
    }
    let mut mid_line = false;
    for k in 0..=chars.len() - w.len() {
        if !chars[k..k + w.len()]
            .iter()
            .map(|x| x.0)
            .eq(w.iter().copied())
        {
            continue;
        }
        let (start, end) = (chars[k].1, chars[k + w.len() - 1].2);
        let before = text[from..start].chars().next_back();
        let after = text[end..to].chars().next();
        if before.is_some_and(char::is_alphanumeric) || after.is_some_and(char::is_alphanumeric) {
            continue;
        }
        let line_start = text[from..start].rfind('\n').map_or(from, |i| from + i + 1);
        if text[line_start..start].chars().any(char::is_alphanumeric) {
            mid_line = true;
            continue;
        }
        let line_end = text[end..to].find('\n').map_or(to, |i| end + i);
        let stop = if text[end..line_end].chars().any(char::is_alphanumeric) {
            end
        } else {
            line_end
        };
        return Some(Ok((line_start, stop)));
    }
    mid_line.then_some(Err(()))
}

/// Localiza a linha pela página e pelo texto citado.
pub fn find(pkg: &Package, page: i32, text: &str) -> Result<Target, String> {
    let wanted = edits::plain(text);
    if wanted.is_empty() {
        return Err("'text' has no letters or digits".into());
    }
    if !pkg.blocks.iter().any(|b| b.page == page) {
        let last = pkg.blocks.iter().map(|b| b.page).max().unwrap_or(0);
        return Err(format!(
            "page {page} has no text in this document (its pages go up to {last})"
        ));
    }
    let editable = |b: &Block| {
        matches!(
            BlockType::of_label(&b.label),
            BlockType::Heading | BlockType::Paragraph
        )
    };
    let base: Vec<&Block> = pkg
        .blocks
        .iter()
        .filter(|b| b.page == page && editable(b))
        .collect();
    let page_pieces: Vec<Block> = pkg.edits.pieces(
        &pkg.blocks
            .iter()
            .filter(|b| b.page == page)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let on_page: Vec<&Block> = page_pieces.iter().filter(|b| editable(b)).collect();

    let found: Vec<String> = on_page
        .iter()
        .filter(|p| edits::plain(&p.text) == wanted)
        .map(|p| p.id.clone())
        .collect();
    if !found.is_empty() {
        return Ok(Target::Pieces(found));
    }

    for i in 0..page_pieces.len() {
        let mut got = String::new();
        let mut ids = Vec::new();
        for p in page_pieces[i..].iter().take_while(|p| editable(p)) {
            got.push_str(&edits::plain(&p.text));
            ids.push(p.id.clone());
            if !wanted.starts_with(&got) {
                break;
            }
            if got == wanted {
                return Ok(Target::Join(ids));
            }
        }
    }

    let mut mid_line: Option<&Block> = None;
    for b in &base {
        for (from, to) in pkg.edits.ranges(b) {
            match locate(&b.text, from, to, &wanted) {
                Some(Ok((start, stop))) => {
                    return Ok(Target::Span {
                        block: b.id.clone(),
                        start,
                        stop,
                    });
                }
                Some(Err(())) => mid_line = mid_line.or(Some(b)),
                None => {}
            }
        }
    }
    if let Some(b) = mid_line {
        return Err(format!(
            "\"{}\" is in the middle of a line of a larger block on page {page}, which starts \"{}\"; a title is a whole line or the start of one, so give the line from its start",
            one_line(text, 80),
            one_line(&b.text, 60)
        ));
    }

    let asked = words(text);
    let mut lines: Vec<(usize, bool, usize, &str)> = on_page
        .iter()
        .map(|b| {
            let mine = words(&b.text);
            let shared = asked
                .iter()
                .filter(|w| mine.iter().any(|m| m.starts_with(w.as_str())))
                .count();
            (
                shared,
                TITLE_LABELS.contains(&b.label.as_str()),
                b.text.len(),
                b.text.as_str(),
            )
        })
        .collect();
    lines.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    let lines: Vec<String> = lines
        .iter()
        .take(8)
        .map(|l| format!("\"{}\"", one_line(l.3, 60)))
        .collect();
    let hint = if lines.is_empty() {
        "That page has no text lines; check the page number.".to_string()
    } else {
        format!("Lines on that page, closest first: {}.", lines.join(", "))
    };
    Err(format!(
        "\"{}\" was not found on page {page}. {hint} Copy the line exactly as 'outline' or 'open' shows it.",
        one_line(text, 80)
    ))
}

pub type Embed<'a> = &'a mut dyn FnMut(&[String]) -> Result<Vec<Vec<f32>>, String>;

/// Aplica a alteração e remonta o pacote, sem gravá-lo; `embed` calcula os vetores dos trechos novos.
pub fn restructure(
    pkg: &Package,
    page: i32,
    text: &str,
    change: Change,
    embed: Embed,
) -> Result<Outcome, String> {
    let target = find(pkg, page, text)?;
    let pieces = match &target {
        Target::Pieces(v) | Target::Join(v) => v.clone(),
        Target::Span { block, start, .. } => vec![edits::piece_id(block, *start)],
    };
    let old_title = pkg
        .edits
        .title()
        .filter(|_| change == Change::Set(Role::Title));
    let mut all = pkg.edits.clone();
    if let Some(e) = old_title {
        let id = e.piece();
        for j in pkg.edits.joined_after(&pkg.blocks, &id) {
            all.remove(&j);
        }
        all.remove(&id);
    }
    let edit = |block: &str, start: usize, stop: Option<usize>, role: Role| Edit {
        block: block.to_string(),
        start,
        stop,
        role,
        at: edits::today(),
    };
    let cut = matches!(target, Target::Span { .. });
    match (&target, change) {
        (Target::Span { .. }, Change::Auto) => {
            return Err(format!(
                "\"{}\" is text inside a larger block and has no fix of its own to undo",
                one_line(text, 80)
            ));
        }
        (Target::Span { .. }, Change::Set(Role::Text)) => {
            return Err(format!(
                "\"{}\" is already text, inside a larger block",
                one_line(text, 80)
            ));
        }
        (Target::Span { block, start, stop }, Change::Set(role)) => {
            let len = pkg
                .blocks
                .iter()
                .find(|b| &b.id == block)
                .map_or(0, |b| b.text.len());
            all.set(edit(block, *start, (*stop < len).then_some(*stop), role));
        }
        (Target::Join(_), Change::Auto) => {
            return Err(format!(
                "\"{}\" is not one line in the table of contents and has no fix of its own to undo",
                one_line(text, 80)
            ));
        }
        (Target::Join(_), Change::Set(Role::Text)) => {
            return Err(format!(
                "\"{}\" is already text; lines are joined only to make a title",
                one_line(text, 80)
            ));
        }
        (Target::Join(v), Change::Set(role)) => {
            for (k, id) in v.iter().enumerate() {
                let (block, start) = edits::split_piece(id);
                let stop = all.get(id).and_then(|e| e.stop);
                all.set(edit(
                    block,
                    start,
                    stop,
                    if k == 0 { role } else { Role::Join },
                ));
            }
        }
        (Target::Pieces(v), Change::Auto) => {
            for id in v {
                for j in pkg.edits.joined_after(&pkg.blocks, id) {
                    all.remove(&j);
                }
                all.remove(id);
            }
        }
        (Target::Pieces(v), Change::Set(role)) => {
            for id in v {
                let (block, start) = edits::split_piece(id);
                let stop = all.get(id).and_then(|e| e.stop);
                all.set(edit(block, start, stop, role));
            }
        }
    }

    let opt = pkg.chunking;
    let built = parser::build(&pkg.blocks, &all, &pkg.tree.source, opt);
    if built.gate.violado > 0 {
        return Err(format!(
            "not applied: with this change the book's own numbering contradicts the table of contents ({}). A numbered title takes its depth from its number: use role=heading for it, or fix the title it belongs under first.",
            built.gate.exemplos.join("; ")
        ));
    }
    let tree = built.tree;

    let mut reembedded = 0;
    let vectors = if pkg.vectors.rows == 0 {
        Tensor::default()
    } else {
        let mut rows = package::carry_vectors(&pkg.tree, &pkg.vectors, &tree);
        let missing: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].is_none()).collect();
        for lote in missing.chunks(64) {
            let texts: Vec<String> = lote
                .iter()
                .map(|&i| tree.embedding_text(&tree.chunks[i]))
                .collect();
            let got = embed(&texts)?;
            if got.len() != lote.len() {
                return Err("the embedding model returned a different number of vectors".into());
            }
            for (&i, v) in lote.iter().zip(got) {
                rows[i] = Some(v);
            }
        }
        reembedded = missing.len();
        Tensor::from_rows(
            rows.into_iter()
                .map(|r| r.expect("embedded above"))
                .collect(),
        )
        .map_err(|e| e.to_string())?
    };

    let shape = |t: &Tree| -> Vec<(i32, i32, String)> {
        t.sections
            .iter()
            .map(|s| (s.parent, s.level, s.block.clone()))
            .collect()
    };
    let unchanged =
        shape(&pkg.tree) == shape(&tree) && pkg.tree.title == tree.title && reembedded == 0;
    Ok(Outcome {
        package: Package {
            embed_model: pkg.embed_model.clone(),
            chunking: opt,
            blocks: pkg.blocks.clone(),
            pages: pkg.pages.clone(),
            tree,
            edits: all,
            vectors,
            edges: built.edges,
        },
        blocks: pieces,
        sections_before: pkg.tree.sections.len() - 1,
        reembedded,
        unchanged,
        cut,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edits::Edits;
    use crate::parser::ParserOptions;
    use crate::tree::Block;

    fn block(id: &str, page: i32, label: &str, text: &str) -> Block {
        Block {
            id: id.into(),
            page,
            label: label.into(),
            text: text.into(),
            ..Default::default()
        }
    }

    fn book() -> Package {
        let blocks = vec![
            block("p1_1", 1, "doc_title", "1 Introduction"),
            block("p1_2", 1, "text", "Intro text about semiconductors."),
            block("p1_3", 1, "paragraph_title", "1.1 Background"),
            block("p1_4", 1, "text", "Background on atoms and electrons."),
            block("p2_1", 2, "paragraph_title", "VISIT THE WEBSITE"),
            block("p2_2", 2, "text", "More background on crystals."),
            block("p2_3", 2, "text", "Diodes and Applications"),
            block("p2_4", 2, "text", "A diode conducts in one direction."),
            block("p3_1", 3, "doc_title", "2 Transistors"),
            block("p3_2", 3, "text", "A transistor amplifies."),
            block(
                "p3_3",
                3,
                "text",
                "3 Amplifiers\n\nAn amplifier makes small signals larger.",
            ),
            block(
                "p3_4",
                3,
                "text",
                "Some closing words.\n4 Oscillators\nAn oscillator makes a signal.",
            ),
            block("p4_1", 4, "paragraph_title", "Chapter 5"),
            block("p4_2", 4, "paragraph_title", "Filters"),
            block("p4_3", 4, "text", "A filter removes some frequencies."),
        ];
        let opt = ParserOptions::default();
        let built = parser::build(&blocks, &Edits::default(), "book", opt);
        let vectors = Tensor::from_rows(vec![vec![1.0, 0.0]; built.tree.chunks.len()]).unwrap();
        Package {
            embed_model: "test".into(),
            chunking: opt,
            blocks,
            pages: (1..=4).map(|p| (p, (1000, 1400))).collect(),
            tree: built.tree,
            edits: Edits::default(),
            vectors,
            edges: built.edges,
        }
    }

    fn run(pkg: &Package, page: i32, text: &str, change: Change) -> Result<Outcome, String> {
        restructure(pkg, page, text, change, &mut |t: &[String]| {
            Ok(vec![vec![0.0, 1.0]; t.len()])
        })
    }

    fn opens(pkg: &Package, block: &str) -> bool {
        pkg.tree.sections.iter().any(|s| s.block == block)
    }

    #[test]
    fn text_removes_a_false_title() {
        let pkg = book();
        assert!(opens(&pkg, "p2_1"));
        let out = run(&pkg, 2, "visit the website", Change::Set(Role::Text)).unwrap();
        assert!(!opens(&out.package, "p2_1"));
        assert!(matches!(
            place(&out.package.tree, &out.blocks),
            Place::Inside(_)
        ));
        assert_eq!(out.package.edits.0.len(), 1);
        assert_eq!(out.package.vectors.rows, out.package.tree.chunks.len());
    }

    #[test]
    fn chapter_promotes_text_and_auto_undoes_it() {
        let pkg = book();
        assert!(!opens(&pkg, "p2_3"));
        let out = run(
            &pkg,
            2,
            "Diodes and Applications",
            Change::Set(Role::Chapter),
        )
        .unwrap();
        let t = &out.package.tree;
        let s = t
            .sections
            .iter()
            .find(|s| s.block == "p2_3")
            .expect("opens a section");
        assert_eq!((s.level, s.parent), (1, 0));
        assert!(out.reembedded > 0);

        let back = run(&out.package, 2, "Diodes and Applications", Change::Auto).unwrap();
        assert!(!opens(&back.package, "p2_3"));
        assert!(back.package.edits.0.is_empty());
    }

    #[test]
    fn same_role_changes_nothing() {
        let pkg = book();
        let out = run(&pkg, 1, "1.1 Background", Change::Set(Role::Heading)).unwrap();
        assert!(out.unchanged);
        assert_eq!(out.reembedded, 0);
    }

    #[test]
    fn errors_say_what_to_do() {
        let pkg = book();
        let err = run(
            &pkg,
            2,
            "Quantum Chromodynamics",
            Change::Set(Role::Heading),
        )
        .err()
        .unwrap();
        assert!(err.contains("VISIT THE WEBSITE"), "{err}");
        let err = run(
            &pkg,
            2,
            "background on crystals",
            Change::Set(Role::Heading),
        )
        .err()
        .unwrap();
        assert!(err.contains("middle of a line"), "{err}");
        let err = run(&pkg, 9, "anything", Change::Set(Role::Heading))
            .err()
            .unwrap();
        assert!(err.contains("up to 4"), "{err}");
        let err = run(
            &pkg,
            2,
            "Diodes and Aplications",
            Change::Set(Role::Chapter),
        )
        .err()
        .unwrap();
        assert!(
            err.contains("closest first: \"Diodes and Applications\""),
            "{err}"
        );
        let err = run(&pkg, 3, "2 Transistor", Change::Set(Role::Chapter))
            .err()
            .unwrap();
        assert!(err.contains("closest first: \"2 Transistors\""), "{err}");
    }

    #[test]
    fn a_title_glued_to_text_is_cut_out_and_auto_joins_it_back() {
        let pkg = book();
        let out = run(&pkg, 3, "3 Amplifiers", Change::Set(Role::Chapter)).unwrap();
        assert!(out.cut);
        let t = &out.package.tree;
        let s = t
            .sections
            .iter()
            .find(|s| s.title == "3 Amplifiers")
            .expect("opens a chapter");
        assert_eq!((s.level, s.parent, s.block.as_str()), (1, 0, "p3_3"));
        let rest = t
            .chunks
            .iter()
            .find(|c| c.text.contains("amplifier makes"))
            .unwrap();
        assert_eq!(rest.section, s.id);
        assert!(
            rest.blocks.iter().any(|b| b.starts_with("p3_3@")),
            "{:?}",
            rest.blocks
        );
        let text = |p: &Package| p.blocks.iter().map(|b| b.text.clone()).collect::<Vec<_>>();
        assert_eq!(text(&out.package), text(&pkg));

        let path = std::env::temp_dir().join(format!("fenda_cut_{}.fnd", std::process::id()));
        package::write_package(&path, &out.package).unwrap();
        let read = package::read_package(&path).unwrap();
        assert_eq!(
            read.tree.to_json().unwrap(),
            out.package.tree.to_json().unwrap()
        );
        let _ = std::fs::remove_file(&path);

        let back = run(&out.package, 3, "3 Amplifiers", Change::Auto).unwrap();
        assert!(back.package.edits.0.is_empty());
        assert!(
            !back
                .package
                .tree
                .sections
                .iter()
                .any(|s| s.title == "3 Amplifiers")
        );
    }

    #[test]
    fn a_title_in_the_middle_of_a_block_splits_it_in_three() {
        let pkg = book();
        let out = run(&pkg, 3, "4 Oscillators", Change::Set(Role::Chapter)).unwrap();
        let t = &out.package.tree;
        let s = t
            .sections
            .iter()
            .find(|s| s.title == "4 Oscillators")
            .expect("opens a chapter");
        assert!(s.block.starts_with("p3_4@"), "{}", s.block);
        let before = t
            .chunks
            .iter()
            .find(|c| c.text.contains("closing words"))
            .unwrap();
        let after = t
            .chunks
            .iter()
            .find(|c| c.text.contains("oscillator makes"))
            .unwrap();
        assert_ne!(before.section, s.id);
        assert_eq!(after.section, s.id);
        assert!(!after.text.contains("4 Oscillators"));
    }

    #[test]
    fn a_title_split_in_two_blocks_is_joined_and_auto_splits_it() {
        let pkg = book();
        let out = run(&pkg, 4, "Chapter 5 Filters", Change::Set(Role::Chapter)).unwrap();
        let t = &out.package.tree;
        let s = t
            .sections
            .iter()
            .find(|s| s.title == "Chapter 5 Filters")
            .expect("one chapter");
        assert_eq!((s.level, s.parent, s.block.as_str()), (1, 0, "p4_1"));
        assert!(!opens(&out.package, "p4_2"));
        assert!(
            !t.chunks
                .iter()
                .any(|c| c.blocks.iter().any(|b| b == "p4_2"))
        );

        let back = run(&out.package, 4, "Chapter 5 Filters", Change::Auto).unwrap();
        assert!(back.package.edits.0.is_empty());
        assert!(
            !back
                .package
                .tree
                .sections
                .iter()
                .any(|s| s.title == "Chapter 5 Filters")
        );

        let err = run(&pkg, 4, "Chapter 5 Filters", Change::Set(Role::Text))
            .err()
            .unwrap();
        assert!(err.contains("already text"), "{err}");
    }

    #[test]
    fn the_book_has_one_title_and_auto_gives_back_the_guess() {
        let pkg = book();
        let guess = pkg.tree.title.clone();
        let a = run(&pkg, 4, "Chapter 5 Filters", Change::Set(Role::Title)).unwrap();
        assert_eq!(a.package.tree.title, "Chapter 5 Filters");
        assert_eq!(a.package.tree.sections[0].title, "Chapter 5 Filters");
        assert!(!a.unchanged);
        let b = run(&a.package, 2, "VISIT THE WEBSITE", Change::Set(Role::Title)).unwrap();
        assert_eq!(b.package.tree.title, "VISIT THE WEBSITE");
        assert_eq!(b.package.edits.0.len(), 1);
        assert!(!opens(&b.package, "p2_1"));
        let c = run(&b.package, 2, "VISIT THE WEBSITE", Change::Auto).unwrap();
        assert_eq!(c.package.tree.title, guess);
        assert!(c.package.edits.0.is_empty());
        assert!(Change::parse("join").is_none());
    }

    #[test]
    fn section_ids_survive_a_change() {
        let pkg = book();
        let before = pkg.tree.section_by_key("p3_1").unwrap();
        let out = run(
            &pkg,
            2,
            "Diodes and Applications",
            Change::Set(Role::Chapter),
        )
        .unwrap();
        let t = &out.package.tree;
        let after = t.section_by_key("p3_1").unwrap();
        assert_ne!(before, after);
        assert_eq!(t.sections[after as usize].title, "2 Transistors");
        assert_eq!(t.section_key(after), "p3_1");
    }

    #[test]
    fn package_round_trip_and_a_stale_save_is_refused() {
        let pkg = book();
        let path = std::env::temp_dir().join(format!("fenda_test_{}.fnd", std::process::id()));
        package::write_package(&path, &pkg).unwrap();
        let back = package::read_package(&path).unwrap();
        assert_eq!(back.tree.to_json().unwrap(), pkg.tree.to_json().unwrap());
        assert_eq!(back.blocks.len(), pkg.blocks.len());

        let a = run(&back, 2, "visit the website", Change::Set(Role::Text)).unwrap();
        let b = run(
            &back,
            2,
            "Diodes and Applications",
            Change::Set(Role::Chapter),
        )
        .unwrap();
        package::save_tree(&path, &back.edits, &a.package).unwrap();
        let err = package::save_tree(&path, &back.edits, &b.package)
            .unwrap_err()
            .to_string();
        assert!(err.contains("another session"), "{err}");
        let now = package::read_package(&path).unwrap();
        assert_eq!(now.edits.0.len(), 1);
        assert_eq!(now.vectors.rows, now.tree.chunks.len());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_text_marks_pages() {
        let pkg = book();
        let c = pkg
            .tree
            .chunks
            .iter()
            .find(|c| c.page_first == 2 && c.page_last == 2)
            .unwrap();
        let mut page = 1;
        let t = c.paged_text(&mut page);
        assert!(
            t.starts_with(
                "[p. 2]
"
            ),
            "{t}"
        );
        assert_eq!(t.matches("[p. ").count(), 1);
        assert_eq!(page, 2);
    }
}
