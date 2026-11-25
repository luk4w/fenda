//! Servidor MCP sobre stdio (JSON-RPC 2.0, uma mensagem por linha).
//!
//! Ferramentas: `search`, `outline`, `open`, `restructure` e `remember`. Todas as respostas indicam livro, seção e página, permitindo verificar a fonte.
//!
//! A biblioteca é formada pelos `.fnd` da pasta `.fenda/` do projeto e de `~/.fenda/`, ressincronizada a cada chamada; adicionar ou remover arquivos dispensa reinicialização.
//! Os modelos são carregados sob demanda e liberados após 5 minutos de inatividade.
//! Os PDFs ao lado dos livros são servidos por HTTP em `127.0.0.1`, numa porta livre escolhida pelo sistema; o link `http` preserva o `#page` até o navegador, o que o `file://` não garante no Windows.
//!
//! O stdout é reservado ao protocolo; todo log é emitido em stderr.

use fnd::context;
use fnd::memory::{self, Memory};
use fnd::multi;
use fnd::package;
use fnd::restructure::{self, Change, Place};
use fnd::tree::Chunk;
use models::Device;
use models::retrieve::{self, Models};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

type Res<T> = Result<T, Box<dyn Error>>;

// Inatividade após a qual os modelos são liberados da memória.
const IDLE: Duration = Duration::from_secs(5 * 60);

type Slot = Arc<Mutex<Option<(Models, Instant)>>>;

const MEMORY_FILE: &str = "memory.fnd";

/// Diretórios da biblioteca, em ordem de precedência: `.fenda/` do projeto, depois `~/.fenda/`.
fn libraries() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(d) = cwd
            .ancestors()
            .map(|a| a.join(".fenda"))
            .find(|p| p.is_dir())
        {
            out.push(d);
        }
    }
    if let Some(h) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let d = PathBuf::from(h).join(".fenda");
        let same = |a: &PathBuf| a.canonicalize().ok() == d.canonicalize().ok();
        if d.is_dir() && !out.iter().any(same) {
            out.push(d);
        }
    }
    out
}

const WHERE_BOOKS: &str = "the project's .fenda folder or ~/.fenda";

/// Estado do servidor: livros carregados, memória do projeto e modelos.
pub struct Server {
    docs: Vec<multi::Doc>,
    list_changed: bool,
    memory: Option<Memory>,
    models_dir: Option<PathBuf>,
    models: Slot,
    /// Hash de cada PDF já conferido, válido enquanto a data de modificação do arquivo não mudar.
    pdfs: RefCell<HashMap<PathBuf, (SystemTime, String)>>,
    /// PDFs liberados para o servidor HTTP, pelo nome no endereço (`<livro>.pdf`).
    served: Served,
    /// Porta do servidor HTTP; ausente se ele não pôde ser iniciado.
    port: Option<u16>,
}

type Served = Arc<Mutex<HashMap<String, PathBuf>>>;

/// Inicia o servidor HTTP dos PDFs numa porta livre de `127.0.0.1` e retorna a porta.
fn serve_pdfs(served: Served) -> Option<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).ok()?;
    let port = listener.local_addr().ok()?.port();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let served = Arc::clone(&served);
            std::thread::spawn(move || {
                let _ = answer(stream, &served);
            });
        }
    });
    Some(port)
}

/// Responde a `GET /<livro>.pdf` com o PDF liberado; qualquer outro pedido recebe 404.
fn answer(mut stream: std::net::TcpStream, served: &Served) -> std::io::Result<()> {
    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut header = String::new();
    while reader.read_line(&mut header)? > 2 {
        header.clear();
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let name = decode(
        target
            .trim_start_matches('/')
            .split(['?', '#'])
            .next()
            .unwrap_or(""),
    );
    let path = served.lock().ok().and_then(|m| m.get(&name).cloned());
    let file = path.and_then(|p| std::fs::File::open(p).ok());
    let Some(mut file) = file.filter(|_| method == "GET" || method == "HEAD") else {
        return stream.write_all(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
    };
    let len = file.metadata()?.len();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
    )?;
    if method == "GET" {
        std::io::copy(&mut file, &mut stream)?;
    }
    Ok(())
}

/// Codifica um segmento de endereço; só letras, dígitos e `-._~` ficam como estão.
fn encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Decodifica as sequências `%XX` de um segmento de endereço.
fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = (b[i] == b'%' && i + 2 < b.len())
            .then(|| std::str::from_utf8(&b[i + 1..i + 3]).ok())
            .flatten()
            .and_then(|h| u8::from_str_radix(h, 16).ok());
        match hex {
            Some(v) => {
                out.push(v);
                i += 3;
            }
            None => {
                out.push(b[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn cut(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut i = n;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    format!("{}…", &s[..i])
}

fn pages(c: &Chunk) -> String {
    if c.page_first <= 0 {
        String::new()
    } else if c.page_first == c.page_last || c.page_last <= 0 {
        format!("p. {}", c.page_first)
    } else {
        format!("p. {}-{}", c.page_first, c.page_last)
    }
}

const INSTRUCTIONS: &str = "Compiled books (.fnd). Each book has a table of contents (chapters and sections, built automatically from the page images) and excerpts with page numbers.
- search: where a topic is covered. Start here; it does not depend on the table of contents.
- outline: the chapters, or the subsections of one section.
- open: the full text of one section.
- restructure: fix the table of contents. It is built automatically and some entries can be wrong: a line listed as a section that is not a title, a real title left inside the text or split in two lines, a title at the wrong depth, a wrong name for the book. When you notice one while reading, fix it and go on. It applies at once; role=auto undoes it. Only the table of contents changes, never the text of the book.
- remember: save a decision or a finding in the project memory.
When a result shows a link to the PDF, give it to the user next to the page you cite: it opens the PDF at that page.";

fn text(s: String) -> Value {
    json!({"content": [{"type": "text", "text": s}]})
}

impl Server {
    pub fn new(memory_path: Option<PathBuf>, models_dir: Option<PathBuf>) -> Res<Self> {
        let memory = match memory_path.or_else(|| memory::find_memory_file(false)) {
            Some(p) => match Memory::open(&p) {
                Ok(m) => {
                    eprintln!("[fenda-mcp] memory: {} ({} notes)", p.display(), m.len());
                    Some(m)
                }
                Err(e) => {
                    eprintln!("[fenda-mcp] memory off: {e}");
                    None
                }
            },
            None => None,
        };
        let models: Slot = Arc::new(Mutex::new(None));
        let watch = Arc::clone(&models);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(30));
                let Ok(mut slot) = watch.lock() else { return };
                if slot
                    .as_ref()
                    .is_some_and(|(_, used)| used.elapsed() >= IDLE)
                {
                    *slot = None;
                    eprintln!(
                        "[fenda-mcp] models unloaded after {} min idle",
                        IDLE.as_secs() / 60
                    );
                }
            }
        });
        let served: Served = Arc::new(Mutex::new(HashMap::new()));
        let port = serve_pdfs(Arc::clone(&served));
        match port {
            Some(p) => eprintln!("[fenda-mcp] PDFs served at http://127.0.0.1:{p}"),
            None => eprintln!("[fenda-mcp] could not start the PDF server; page links are off"),
        }
        let mut server = Server {
            docs: Vec::new(),
            list_changed: false,
            memory,
            models_dir,
            models,
            pdfs: RefCell::new(HashMap::new()),
            served,
            port,
        };
        server.sync();
        if server.docs.is_empty() {
            eprintln!("[fenda-mcp] no .fnd in {} yet", WHERE_BOOKS);
        }
        Ok(server)
    }

    /// Sincroniza os livros carregados com o conteúdo dos diretórios da biblioteca.
    fn sync(&mut self) {
        let mut files: Vec<PathBuf> = Vec::new();
        for dir in libraries() {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            let mut here: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.is_file()
                        && p.extension().is_some_and(|x| x == "fnd")
                        && p.file_name().is_some_and(|n| n != MEMORY_FILE)
                })
                .collect();
            here.sort();
            for f in here {
                let name = multi::short_name(&f.to_string_lossy());
                if !files
                    .iter()
                    .any(|g| multi::short_name(&g.to_string_lossy()) == name)
                {
                    files.push(f);
                }
            }
        }
        let before = self.names();
        self.docs.retain(|d| files.contains(&d.path));
        for f in &files {
            if let Some(d) = self.docs.iter_mut().find(|d| &d.path == f) {
                if let Err(e) = d.refresh() {
                    eprintln!("[fenda-mcp] {}: could not reload: {e}", d.name);
                }
                continue;
            }
            match multi::load(&f.to_string_lossy()) {
                Ok(d) => {
                    eprintln!(
                        "[fenda-mcp] {}: {} sections, {} chunks",
                        d.name,
                        d.pkg.tree.sections.len().saturating_sub(1),
                        d.pkg.tree.chunks.len()
                    );
                    self.docs.push(d);
                }
                Err(e) => eprintln!("[fenda-mcp] skipping {}: {e}", f.display()),
            }
        }
        self.docs.sort_by(|a, b| a.name.cmp(&b.name));
        if self.names() != before {
            eprintln!("[fenda-mcp] books: {}", self.names());
            self.list_changed = true;
        }
    }

    /// Executa `f` com os modelos, carregando-os quando necessário.
    fn with_models<T>(&self, f: impl FnOnce(&Models) -> Res<T>) -> Res<T> {
        let all: Vec<usize> = (0..self.docs.len()).collect();
        let model = multi::common_model(&self.docs, &all)?;
        let mut slot = self
            .models
            .lock()
            .map_err(|_| "the models lock is poisoned")?;
        if slot
            .as_ref()
            .is_some_and(|(m, _)| m.embedder.name() != model)
        {
            *slot = None;
        }
        if slot.is_none() {
            let m = Models::load(model, self.models_dir.as_deref(), Device::Cpu)
                .map_err(|e| e.to_string())?;
            *slot = Some((m, Instant::now()));
        }
        let (models, used) = slot.as_mut().expect("loaded above");
        let out = f(models);
        *used = Instant::now();
        out
    }

    /// Processa requisições do stdin até o fim da entrada.
    pub fn run(&mut self) -> Res<()> {
        let stdin = std::io::stdin();
        let mut out = std::io::stdout();
        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let req: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[fenda-mcp] invalid json: {e}");
                    continue;
                }
            };
            if let Some(reply) = self.handle(&req) {
                writeln!(out, "{reply}")?;
                out.flush()?;
            }
            if self.list_changed {
                self.list_changed = false;
                writeln!(
                    out,
                    "{}",
                    json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
                )?;
                out.flush()?;
            }
        }
        Ok(())
    }

    fn handle(&mut self, req: &Value) -> Option<Value> {
        let method = req["method"].as_str().unwrap_or("");
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let reply = |result: Value| -> Option<Value> {
            if id.is_null() {
                None
            } else {
                Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
            }
        };
        match method {
            "initialize" => reply(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": true}},
                "serverInfo": {"name": "fenda", "version": env!("CARGO_PKG_VERSION")},
                "instructions": INSTRUCTIONS,
            })),
            "ping" => reply(json!({})),
            "tools/list" => {
                self.sync();
                self.list_changed = false;
                reply(json!({"tools": self.tools()}))
            }
            "tools/call" => {
                let result = match self.call(&req["params"]) {
                    Ok(v) => v,
                    Err(e) => {
                        json!({"isError": true, "content": [{"type": "text", "text": format!("error: {e}")}]})
                    }
                };
                reply(result)
            }
            m if m.starts_with("notifications/") => None,
            _ => {
                if id.is_null() {
                    None
                } else {
                    Some(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32601, "message": format!("unknown method: {method}")}}))
                }
            }
        }
    }

    fn names(&self) -> String {
        self.docs
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn tools(&self) -> Value {
        let list = self.names();
        json!([
            {
                "name": "search",
                "description": format!("Semantic search over the compiled documents ({list}), ranked together: the best excerpts of all documents, each with its document, path in the hierarchy and page, plus the neighboring sections as an index. Use it to locate where a topic is covered, then use 'open' to read the whole section."),
                "inputSchema": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "what you are looking for, in natural language"},
                    "document": {"type": "string", "description": format!("optional: restrict to one document ({list})")},
                    "k": {"type": "integer", "description": "how many excerpts in total, across all documents (default 5)"}
                }, "required": ["query"]}
            },
            {
                "name": "outline",
                "description": "Browse the document hierarchy: returns the direct subsections of a section (or the chapters, if 'section' is omitted), with id, page and size in tokens. Use it to explore the structure when you do not know what to look for, or to drill down to the right section after 'search'.",
                "inputSchema": {"type": "object", "properties": {
                    "document": {"type": "string", "description": format!("which document ({list})")},
                    "section": {"type": "string", "description": "optional: the id of the section whose subsections to list, as shown in [id ...] (like p42_1); omitted = chapters"}
                }, "required": ["document"]}
            },
            {
                "name": "open",
                "description": "Read the full text of a section, by the id that 'search' or 'outline' returned. Use it when the search excerpt is not enough. A mark like [p. 12] starts the text of each page.",
                "inputSchema": {"type": "object", "properties": {
                    "document": {"type": "string", "description": format!("which document ({list})")},
                    "section": {"type": "string", "description": "the id of the section, as shown in [id ...] (like p42_1)"}
                }, "required": ["document", "section"]}
            },
            {
                "name": "remember",
                "description": "Write a note into the project's memory graph, so a decision, a measurement or a retraction survives this session. The note is APPEND-ONLY and never edited: to correct an earlier note, write a new one and pass its id in 'supersedes'. A superseded note is always returned together with the one that replaced it, so a retracted conclusion cannot be served as if it still held.",
                "inputSchema": {"type": "object", "properties": {
                    "title": {"type": "string", "description": "one line naming the finding"},
                    "body": {"type": "string", "description": "what was decided or measured, and why"},
                    "supersedes": {"type": "integer", "description": "optional: id of the note this one replaces"},
                    "source": {"type": "string", "description": "optional: excerpt copied from where this came from, with its origin. Copied, not referenced: a live reference would make the decision look based on text nobody read."}
                }, "required": ["title", "body"]}
            },
            {
                "name": "restructure",
                "description": "Fix one entry of a book's table of contents. Give the page and the line exactly as 'outline', 'open' or 'search' shows it, and the role that line should have:\n  heading: a numbered title (like \"2.3 Diodes\"); its depth comes from the number;\n  chapter, section, subsection: a title without a number, at that depth;\n  text: not a title, just text inside its section;\n  title: the name of the whole book (one per book);\n  auto: undo your earlier fix of this line.\nA title that the page reading glued to the text around it can be given too: it is cut out of that text (auto joins it back). So can a title the page reading split in two lines, like \"Chapter 1\" and \"Introduction\": give it whole, \"Chapter 1 Introduction\", and the two become one (auto splits them again). The table of contents is rebuilt at once and the answer says where the line ended up. Only the table of contents changes, never the text of the book.",
                "inputSchema": {"type": "object", "properties": {
                    "document": {"type": "string", "description": format!("which document ({list})")},
                    "page": {"type": "integer", "description": "page where the line is"},
                    "text": {"type": "string", "description": "the whole line, as shown"},
                    "role": {"type": "string", "enum": Change::NAMES, "description": "what the line is: heading, chapter, section, subsection, text, title, or auto to undo"}
                }, "required": ["document", "page", "text", "role"]}
            }
        ])
    }

    fn call(&mut self, params: &Value) -> Res<Value> {
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        self.sync();
        if self.docs.is_empty() {
            return Err(format!("no books: put .fnd files in {}", WHERE_BOOKS).into());
        }
        match params["name"].as_str().unwrap_or("") {
            "search" => self.search(&args),
            "outline" => self.outline(&args),
            "open" => self.open(&args),
            "remember" => self.remember(&args),
            "restructure" => self.restructure(&args),
            other => Err(format!("unknown tool: {other}").into()),
        }
    }

    fn doc(&self, args: &Value) -> Res<&multi::Doc> {
        let name = args["document"].as_str().ok_or("missing 'document'")?;
        self.docs.iter().find(|d| d.name == name).ok_or_else(|| {
            format!("document '{name}' not found; available: {}", self.names()).into()
        })
    }

    /// Link para o PDF de mesmo nome ao lado do livro, somente se o hash coincidir com o gravado na compilação.
    fn pdf_link(&self, d: &multi::Doc) -> Option<String> {
        let port = self.port?;
        let sha = d.pkg.source_sha256.as_ref()?;
        let pdf = std::path::absolute(d.path.with_extension("pdf")).ok()?;
        let modified = std::fs::metadata(&pdf).and_then(|m| m.modified()).ok()?;
        let mut cache = self.pdfs.borrow_mut();
        let hash = match cache.get(&pdf) {
            Some((m, h)) if *m == modified => h.clone(),
            _ => {
                let h = crate::install::sha256(&pdf).ok()?;
                cache.insert(pdf.clone(), (modified, h.clone()));
                h
            }
        };
        if hash != *sha {
            return None;
        }
        let name = format!("{}.pdf", d.name);
        let url = format!("http://127.0.0.1:{port}/{}", encode(&name));
        self.served.lock().ok()?.insert(name, pdf);
        Some(url)
    }

    fn search(&mut self, args: &Value) -> Res<Value> {
        let query = args["query"].as_str().ok_or("missing 'query'")?.to_string();
        let k = args["k"].as_u64().unwrap_or(5).max(1) as usize;
        let filter = args["document"].as_str().unwrap_or("");
        let selected: Vec<usize> = (0..self.docs.len())
            .filter(|&i| filter.is_empty() || self.docs[i].name == filter)
            .collect();
        if selected.is_empty() {
            return Err(
                format!("document '{filter}' not found; available: {}", self.names()).into(),
            );
        }
        let (q, hits) =
            self.with_models(|m| Ok(retrieve::search(&self.docs, &selected, m, &query, k)?))?;
        let mut out = String::new();
        for di in multi::docs_in_order(&hits) {
            let d = &self.docs[di];
            let t = &d.pkg.tree;
            let doc_hits = multi::hits_of(&hits, di);
            let ctx = context::expand_context(t, &d.pkg.edges, &d.pkg.vectors, &q, &doc_hits);
            let from_vector: HashSet<usize> = doc_hits.iter().map(|h| h.chunk).collect();
            let hit_sections: HashSet<i32> =
                doc_hits.iter().map(|h| t.chunks[h.chunk].section).collect();
            let near = adjacent(t, &d.pkg.edges, &hit_sections);
            let link = self.pdf_link(d);

            out.push_str(&format!(
                "## {} — {}\n\n",
                d.name,
                t.title.replace('\n', " ")
            ));
            let mut neighbors = String::new();
            let mut listed: HashSet<i32> = HashSet::new();
            for &cid in &ctx {
                let c = &t.chunks[cid];
                let p = pages(c);
                let p = if p.is_empty() {
                    String::new()
                } else {
                    format!("  ({p})")
                };
                if from_vector.contains(&cid) {
                    let subs = t.sections.iter().filter(|o| o.parent == c.section).count();
                    let subs = if subs == 0 {
                        String::new()
                    } else {
                        format!("  [{subs} subsections: 'outline' lists them]")
                    };
                    let body = c.paged_text(&mut c.page_first.clone());
                    let url = match &link {
                        Some(u) if c.page_first > 0 => format!("{u}#page={}\n", c.page_first),
                        _ => String::new(),
                    };
                    out.push_str(&format!(
                        "### [id {}] {}{p}{subs}\n{url}{}\n\n",
                        t.section_key(c.section),
                        t.location(c).replace('\n', " "),
                        cut(&body, 500)
                    ));
                } else if !hit_sections.contains(&c.section)
                    && near.contains(&c.section)
                    && listed.insert(c.section)
                {
                    let parent = t.sections[c.section as usize].parent;
                    let sister = hit_sections
                        .iter()
                        .copied()
                        .filter(|&h| t.sections[h as usize].parent == parent)
                        .min_by_key(|&h| (h - c.section).abs());
                    let name = match sister {
                        Some(h) => format!(
                            "{} (beside id {})",
                            t.sections[c.section as usize].title,
                            t.section_key(h)
                        ),
                        None => t.breadcrumb(c.section),
                    };
                    neighbors.push_str(&format!(
                        "  [id {}] {}{p}  ~{} tok\n",
                        t.section_key(c.section),
                        name.replace('\n', " "),
                        d.section_tokens(c.section)
                    ));
                }
            }
            if !neighbors.is_empty() {
                out.push_str(&format!(
                    "Neighbors in the hierarchy (use 'open' to read):\n{neighbors}\n"
                ));
            }
        }
        if filter.is_empty() {
            out.push_str(&self.search_memory(&q, k));
        }
        Ok(text(out))
    }

    fn search_memory(&self, q: &[f32], k: usize) -> String {
        let Some(m) = &self.memory else {
            return String::new();
        };
        if m.len() == 0 {
            return String::new();
        }
        let mut out = String::new();
        for (id, _) in m.search(q, k) {
            let Some(n) = m.find(id) else { continue };
            let newer = m.superseded_by(id);
            out.push_str(&format!("### [note {id}] {}", n.title));
            if !newer.is_empty() {
                out.push_str("   ** SUPERSEDED **");
            }
            out.push_str(&format!("\n{}\n", cut(&n.body, 400)));
            for nid in newer {
                if let Some(s) = m.find(nid) {
                    out.push_str(&format!(
                        "  -> superseded by [note {nid}] {}\n  {}\n",
                        s.title,
                        cut(&s.body, 300)
                    ));
                }
            }
            out.push('\n');
        }
        if out.is_empty() {
            out
        } else {
            format!("## memory\n\n{out}")
        }
    }

    fn outline(&self, args: &Value) -> Res<Value> {
        let d = self.doc(args)?;
        let t = &d.pkg.tree;
        let parent = match &args["section"] {
            Value::Null => 0,
            v => section_arg(d, v)?,
        };
        let head = if parent == 0 {
            t.title.clone()
        } else {
            t.breadcrumb(parent)
        }
        .replace('\n', " ");
        let mut out = format!("# {} — {head}\n\n", d.name);
        let mut n = 0;
        for s in t
            .sections
            .iter()
            .filter(|s| s.id != 0 && s.parent == parent)
        {
            n += 1;
            out.push_str(&format!("  [id {}] {}", s.block, s.title));
            let pf = d.first_page(s.id);
            if pf > 0 {
                out.push_str(&format!("  (p. {pf})"));
            }
            let tok = d.section_tokens(s.id);
            if tok > 0 {
                out.push_str(&format!("  ~{tok} tok"));
            }
            if t.sections.iter().any(|f| f.parent == s.id) {
                out.push_str("  [has subsections]");
            }
            out.push('\n');
        }
        if n == 0 {
            out.push_str("(no subsections)\n");
        }
        Ok(text(out))
    }

    fn open(&self, args: &Value) -> Res<Value> {
        let d = self.doc(args)?;
        let t = &d.pkg.tree;
        let sec = section_arg(d, &args["section"])?;
        let mut out = format!(
            "# {}\n({}, id {})\n\n",
            t.breadcrumb(sec).replace('\n', " "),
            d.name,
            t.section_key(sec)
        );
        let (mut pf, mut pl, mut found) = (0, 0, false);
        let mut page = 0;
        for c in t.chunks.iter().filter(|c| c.section == sec) {
            found = true;
            out.push_str(&c.paged_text(&mut page));
            out.push_str("\n\n");
            if c.page_first > 0 && (pf == 0 || c.page_first < pf) {
                pf = c.page_first;
            }
            pl = pl.max(c.page_last);
        }
        if !found {
            out.push_str("(section with no text of its own - a heading that groups subsections)\n");
        }
        if pf > 0 {
            out.push_str(&format!("- pages {pf}-{pl} of {}\n", d.name));
            if let Some(u) = self.pdf_link(d) {
                out.push_str(&format!("  {u}#page={pf}\n"));
            }
        }
        let children: Vec<String> = t
            .sections
            .iter()
            .filter(|s| s.parent == sec)
            .map(|s| format!("  [id {}] {}", s.block, s.title))
            .collect();
        if !children.is_empty() {
            out.push_str(&format!("\nSubsections:\n{}\n", children.join("\n")));
        }
        Ok(text(out))
    }

    fn remember(&mut self, args: &Value) -> Res<Value> {
        let title = args["title"].as_str().ok_or("missing 'title'")?;
        let body = args["body"].as_str().ok_or("missing 'body'")?;
        let source = args["source"].as_str().unwrap_or("");
        let supersedes = args["supersedes"].as_u64().unwrap_or(0);

        if self.memory.is_none() {
            let p = memory::find_memory_file(true)
                .ok_or("no project root found (.fenda or .git) to hold the memory")?;
            self.memory = Some(Memory::open(&p)?);
        }
        let full = if source.is_empty() {
            body.to_string()
        } else {
            format!("{body}\n\n--- source ---\n{source}")
        };
        let vector = self.with_models(|m| {
            Ok(m.embedder
                .embed_passages(&[format!("{title}\n{full}")])?
                .remove(0))
        })?;
        let m = self.memory.as_mut().unwrap();
        if supersedes != 0 && m.find(supersedes).is_none() {
            return Err(format!("note {supersedes} does not exist").into());
        }
        let links: Vec<(u64, u32)> = if supersedes != 0 {
            vec![(supersedes, memory::SUPERSEDES)]
        } else {
            vec![]
        };
        let id = m.add(title, &full, &vector, &links)?;
        let mut out = format!("note {id} written to {}", m.path().display());
        if supersedes != 0 {
            out.push_str(&format!("; supersedes note {supersedes}"));
        }
        out.push_str(&format!(" ({} notes total)", m.len()));
        Ok(text(out))
    }

    fn restructure(&mut self, args: &Value) -> Res<Value> {
        let name = args["document"].as_str().ok_or("missing 'document'")?;
        let di = self
            .docs
            .iter()
            .position(|d| d.name == name)
            .ok_or_else(|| format!("document '{name}' not found; available: {}", self.names()))?;
        let page = args["page"].as_i64().ok_or("missing 'page'")? as i32;
        let line = args["text"].as_str().ok_or("missing 'text'")?.trim();
        let role = args["role"].as_str().ok_or("missing 'role'")?;
        let change = Change::parse(role)
            .ok_or_else(|| format!("'role' must be one of: {}", Change::NAMES.join(", ")))?;

        let out = self.with_models(|models| {
            let mut embed =
                |t: &[String]| models.embedder.embed_passages(t).map_err(|e| e.to_string());
            Ok(restructure::restructure(
                &self.docs[di].pkg,
                page,
                line,
                change,
                &mut embed,
            )?)
        })?;
        let path = self.docs[di].path.clone();
        package::save_tree(&path, &self.docs[di].pkg.edits, &out.package)?;
        self.docs[di].replace(out.package)?;

        let t = &self.docs[di].pkg.tree;
        let crumb = |s: i32| t.breadcrumb(s).replace('\n', " ");
        let place = match restructure::place(t, &out.blocks) {
            Place::Opens(s) => {
                let subs = t.sections.iter().filter(|o| o.parent == s).count();
                format!(
                    "now [id {}] {} ({subs} subsections)",
                    t.section_key(s),
                    crumb(s)
                )
            }
            Place::Inside(s) => format!("now text inside [id {}] {}", t.section_key(s), crumb(s)),
            Place::Nowhere => "now outside every section".to_string(),
        };
        let mut msg = if change == Change::Set(fnd::edits::Role::Title) {
            format!(
                "p. {page} \"{line}\" -> title: the book is now called \"{}\".",
                t.title
            )
        } else {
            format!("p. {page} \"{line}\" -> {}: {place}.", change.name())
        };
        if out.cut {
            msg.push_str(" It was cut out of the text around it.");
        }
        if out.unchanged {
            msg.push_str(" It already was; the table of contents did not change.");
        } else {
            msg.push_str(&format!(
                "\nTable of contents: {} -> {} sections; {} excerpts re-embedded.",
                out.sections_before,
                t.sections.len() - 1,
                out.reembedded
            ));
        }
        Ok(text(msg))
    }
}

/// Id da seção a partir do que o modelo enviou: aceita o rótulo inteiro (`[id p42_1]`, `section p42_1`).
fn bare_id(key: &str) -> &str {
    let k = key
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    k.split_whitespace().last().unwrap_or(k)
}

fn section_arg(d: &multi::Doc, v: &Value) -> Res<i32> {
    let key = v
        .as_str()
        .ok_or("missing 'section': give the id shown in [id ...], like p42_1")?;
    d.pkg.tree.section_by_key(bare_id(key)).ok_or_else(|| {
        format!("section '{key}' not found in {}; use the id that 'search' or 'outline' shows in [id ...], like p42_1", d.name).into()
    })
}

fn adjacent(t: &fnd::tree::Tree, edges: &[fnd::tree::Edge], hits: &HashSet<i32>) -> HashSet<i32> {
    let mut near = HashSet::new();
    for &s in hits {
        let parent = t.sections[s as usize].parent;
        if parent > 0 {
            near.insert(parent);
        }
        let sisters: Vec<i32> = t
            .sections
            .iter()
            .filter(|o| o.id != 0 && o.parent == parent)
            .map(|o| o.id)
            .collect();
        if let Some(i) = sisters.iter().position(|&o| o == s) {
            near.extend(i.checked_sub(1).map(|j| sisters[j]));
            near.extend(sisters.get(i + 1).copied());
        }
        for e in edges {
            if e.from == s {
                near.insert(e.to);
            } else if e.to == s {
                near.insert(e.from);
            }
        }
    }
    near
}
