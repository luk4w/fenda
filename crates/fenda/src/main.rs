//! Leitor de pacotes `.fnd`: busca semântica, perguntas com citação e servidor MCP. A compilação de livros não faz parte deste programa.

mod chat;
mod install;
mod mcp;

use clap::{Args, Parser as ClapParser, Subcommand};
use fnd::context;
use fnd::multi;
use fnd::tree::{Chunk, Tree};
use models::Device;
use models::retrieve::{self, Models};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(ClapParser)]
#[command(
    name = "fenda",
    about = "fenda - reads .fnd knowledge packages (search, ask, MCP)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// semantic search in one or more .fnd
    Search(SearchArgs),
    /// question answered from one or more .fnd, with cited sections
    Ask(AskArgs),
    /// MCP server (stdio) over the books of the project's .fenda and of ~/.fenda
    Mcp(McpArgs),
    /// put fenda in ~/.fenda, download its models and register the MCP server in Claude Code, Codex and Gemini (once per machine)
    Install,
    /// undo `install`; the books stay
    Uninstall,
}

#[derive(Args)]
struct ModelArgs {
    /// models directory (default: $FENDA_MODELS, then .fenda/models, then ~/.fenda/models)
    #[arg(long)]
    models: Option<String>,
}

impl ModelArgs {
    fn dir(&self) -> Option<&Path> {
        self.models.as_deref().map(Path::new)
    }

    fn load(&self, docs: &[multi::Doc]) -> Result<Models, Box<dyn std::error::Error>> {
        let all: Vec<usize> = (0..docs.len()).collect();
        let model = multi::common_model(docs, &all)?;
        Models::load(model, self.dir(), Device::Cpu)
    }
}

#[derive(Args)]
struct McpArgs {
    /// project memory file (default: .fenda/memory.fnd found from the working directory up)
    #[arg(long)]
    memory: Option<String>,
    #[command(flatten)]
    m: ModelArgs,
}

#[derive(Args)]
struct AskArgs {
    /// one or more .fnd files, then the question
    #[arg(required = true, num_args = 2..)]
    inputs: Vec<String>,
    /// how many hits seed the context
    #[arg(short, default_value_t = 5)]
    k: usize,
    /// chat model
    #[arg(short, default_value = "gemma4:e4b-it-qat")]
    model: String,
    /// chat server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// chat server port
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[command(flatten)]
    m: ModelArgs,
}

#[derive(Args)]
struct SearchArgs {
    /// one or more .fnd files, then the query
    #[arg(required = true, num_args = 2..)]
    inputs: Vec<String>,
    /// how many results
    #[arg(short, default_value_t = 5)]
    k: usize,
    /// headers only
    #[arg(short, long)]
    quiet: bool,
    #[command(flatten)]
    m: ModelArgs,
}

fn cmd_search(a: &SearchArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let (files, query) = multi::split_inputs(&a.inputs)?;
    let docs = files
        .iter()
        .map(|f| multi::load(f))
        .collect::<Result<Vec<_>, _>>()?;
    let all: Vec<usize> = (0..docs.len()).collect();
    let models = a.m.load(&docs)?;
    let (_, hits) = retrieve::search(&docs, &all, &models, query, a.k)?;
    for (i, h) in hits.iter().enumerate() {
        let t = &docs[h.doc].pkg.tree;
        let c = &t.chunks[h.chunk];
        let doc = if docs.len() > 1 {
            format!("{}: ", docs[h.doc].name)
        } else {
            String::new()
        };
        let score = match h.rerank {
            Some(r) => format!("rerank={r:.3} cos={:.4}", h.score),
            None => format!("score={:.4}", h.score),
        };
        println!(
            "#{}  {score}  [{doc}{}]",
            i + 1,
            t.location(c).replace('\n', " ")
        );
        if !a.quiet {
            let previa: String = c.text.chars().take(400).collect();
            let resto = if c.text.chars().count() > 400 {
                "…"
            } else {
                ""
            };
            println!("{previa}{resto}");
        }
        println!();
    }
    Ok(ExitCode::SUCCESS)
}

fn chunk_header(t: &Tree, c: &Chunk, doc: Option<&str>) -> String {
    let crumb = match doc {
        Some(d) => format!("{d}: {}", t.location(c).replace('\n', " ")),
        None => t.location(c).replace('\n', " "),
    };
    if c.page_first <= 0 {
        format!("[{crumb}]")
    } else if c.page_first == c.page_last || c.page_last <= 0 {
        format!("[{crumb}, p. {}]", c.page_first)
    } else {
        format!("[{crumb}, p. {}-{}]", c.page_first, c.page_last)
    }
}

fn cmd_ask(a: &AskArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let (files, question) = multi::split_inputs(&a.inputs)?;
    let docs = files
        .iter()
        .map(|f| multi::load(f))
        .collect::<Result<Vec<_>, _>>()?;
    let all: Vec<usize> = (0..docs.len()).collect();
    let models = a.m.load(&docs)?;
    let (query, hits) = retrieve::search(&docs, &all, &models, question, a.k)?;

    let several = docs.len() > 1;
    let mut excerpts = String::new();
    let mut sent: u64 = 0;
    for d in multi::docs_in_order(&hits) {
        let p = &docs[d].pkg;
        let context = context::expand_context(
            &p.tree,
            &p.edges,
            &p.vectors,
            &query,
            &multi::hits_of(&hits, d),
        );
        for &chunk in &context {
            let c = &p.tree.chunks[chunk];
            let name = several.then_some(docs[d].name.as_str());
            excerpts.push_str(&format!(
                "{}\n{}\n\n",
                chunk_header(&p.tree, c, name),
                c.text
            ));
            sent += c.tokens_est.max(0) as u64;
        }
    }

    let titles: Vec<String> = docs
        .iter()
        .map(|d| format!("\"{}\" ({})", d.pkg.tree.title.replace('\n', " "), d.name))
        .collect();
    let system = format!(
        "You answer questions about {} {} using ONLY the provided excerpts. Each excerpt is \
         prefixed in brackets by {}its section path and page. Whenever you use information from \
         an excerpt, cite {}its section and page in your answer; never invent citations. If the \
         excerpts do not contain the answer, say so explicitly. Reply in the same language as \
         the question.",
        if several {
            "the documents"
        } else {
            "the document"
        },
        titles.join(", "),
        if several { "the document name, " } else { "" },
        if several { "the document, " } else { "" },
    );
    let user = format!("Excerpts:\n\n{excerpts}\nQuestion: {question}");
    let server = chat::ChatClient::new(&a.host, a.port);
    let (answer, read_tokens) = server.chat(&a.model, &system, &user)?;

    if let Some(read) = read_tokens {
        if read * 2 < sent {
            eprintln!(
                "warning: the chat server read about {read} tokens of a context of at least {sent}; \
                 its context window is too small and the answer below is missing excerpts. Raise \
                 the window (Ollama: set OLLAMA_CONTEXT_LENGTH before `ollama serve`) and ask again.\n"
            );
        }
    }
    println!("{answer}");

    Ok(ExitCode::SUCCESS)
}

fn cmd_mcp(a: &McpArgs) -> Result<ExitCode, Box<dyn std::error::Error>> {
    mcp::Server::new(
        a.memory.as_ref().map(PathBuf::from),
        a.m.models.as_ref().map(PathBuf::from),
    )?
    .run()?;
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match &cli.command {
        Command::Search(a) => cmd_search(a),
        Command::Ask(a) => cmd_ask(a),
        Command::Mcp(a) => cmd_mcp(a),
        Command::Install => install::install().map(|()| ExitCode::SUCCESS),
        Command::Uninstall => install::uninstall().map(|()| ExitCode::SUCCESS),
    };
    match r {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
