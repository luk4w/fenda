//! Instalação e remoção do leitor no diretório do usuário.
//!
//! `install` copia o executável para `~/.fenda/bin`, obtém os modelos do Release e registra o servidor MCP no Claude Code, no Codex e no Antigravity. `uninstall` reverte essas etapas e preserva os livros.
//!
//! Os modelos distribuídos são os mesmos usados na compilação dos livros, com integridade verificada por SHA-256.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

type Res<T> = Result<T, Box<dyn Error>>;

// Base dos Releases; cada modelo tem um Release próprio, de tag `<pasta>-<hash>`, com os arquivos nos nomes originais.
const RELEASES: &str = "https://github.com/luk4w/fenda/releases/download";

// Arquivos de modelo exigidos pelo leitor: (pasta, arquivo, tamanho em bytes, SHA-256).
const FILES: [(&str, &str, u64, &str); 6] = [
    (
        "bge-m3",
        "model.int8.onnx",
        569_872_326,
        "63e806d9763c82e8ffb55e32d577a64cacf2a57be1ac0d803f118cf770cb399d",
    ),
    (
        "bge-m3",
        "tokenizer.json",
        17_082_821,
        "6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790",
    ),
    (
        "bge-m3",
        "fenda.json",
        180,
        "98f7c961c32a984ab671d8832d60d3eeff92ad20c7dff5268e682619c10d5484",
    ),
    (
        "mmarco-mMiniLMv2-L12-H384-v1",
        "model.int8.onnx",
        118_484_559,
        "6aa96ed85ae35148ddcab98558cde21c7fb60d757d01414fdc71143abd645aa0",
    ),
    (
        "mmarco-mMiniLMv2-L12-H384-v1",
        "tokenizer.json",
        17_082_734,
        "3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20",
    ),
    (
        "mmarco-mMiniLMv2-L12-H384-v1",
        "fenda.json",
        130,
        "d225d5a827e352e31c6bdb5a494e0550f65c09b327452c1a25e319adb7d8bbe1",
    ),
];

/// Tag do Release de um modelo, derivada dos hashes dos seus arquivos; qualquer alteração no modelo gera uma tag nova.
pub fn model_tag(folder: &str) -> String {
    let mut h = Sha256::new();
    for (f, file, _, sha) in FILES {
        if f == folder {
            h.update(format!("{file} {sha}\n"));
        }
    }
    format!("{folder}-{}", &format!("{:x}", h.finalize())[..12])
}

fn user_dir() -> Res<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .ok_or_else(|| "no USERPROFILE or HOME".into())
}

pub fn home() -> Res<PathBuf> {
    Ok(user_dir()?.join(".fenda"))
}

pub fn sha256(path: &Path) -> Res<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn mb(n: u64) -> String {
    format!("{:.0} MB", n as f64 / 1_048_576.0)
}

/// Baixa para um arquivo `.partial` e o move para o destino somente após validar tamanho e hash.
fn download(url: &str, dest: &Path, size: u64, sha: &str) -> Res<()> {
    let partial = PathBuf::from(format!("{}.partial", dest.display()));
    let mut body = ureq::get(url)
        .call()
        .map_err(|e| format!("could not download {url}: {e}"))?
        .into_reader();
    let mut out = fs::File::create(&partial)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let (mut got, mut shown) = (0u64, 0u64);
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        h.update(&buf[..n]);
        got += n as u64;
        if size > 10 << 20 && got * 10 / size > shown {
            shown = got * 10 / size;
            eprint!(".");
        }
    }
    out.sync_all()?;
    drop(out);
    let hex = format!("{:x}", h.finalize());
    if got != size || hex != sha {
        let _ = fs::remove_file(&partial);
        return Err(
            format!("{url} arrived different from the release ({got} bytes); try again").into(),
        );
    }
    fs::rename(&partial, dest)?;
    Ok(())
}

/// Clientes registrados pela própria linha de comando: (nome, executável, argumentos de escopo do usuário).
const CLIS: [(&str, &str, &[&str]); 2] = [
    ("Claude Code", "claude", &["-s", "user"]),
    ("Codex", "codex", &[]),
];

/// Executa a linha de comando de um cliente; no Windows ela pode ser `.cmd` ou `.exe`.
fn run_cli(bin: &str, args: &[&str]) -> Option<std::process::Output> {
    [bin.to_string(), format!("{bin}.cmd"), format!("{bin}.exe")]
        .iter()
        .find_map(|c| Command::new(c).args(args).output().ok())
}

/// Registra o servidor no escopo do usuário, substituindo um registro anterior.
fn register_cli(name: &str, bin: &str, scope: &[&str], exe: &str) -> String {
    let _ = run_cli(bin, &[&["mcp", "remove", "fenda"][..], scope].concat());
    let add = [&["mcp", "add"][..], scope, &["fenda", "--", exe, "mcp"][..]].concat();
    let manual = format!("{bin} {} -- \"{exe}\" mcp", add[..add.len() - 3].join(" "));
    match run_cli(bin, &add) {
        Some(o) if o.status.success() => format!("{name}: registered for every project"),
        Some(o) => format!(
            "{name}: could not register ({}); run: {manual}",
            String::from_utf8_lossy(&o.stderr).trim()
        ),
        None => format!("{name}: not found; if you install it, run: {manual}"),
    }
}

/// Clientes configurados por arquivo JSON com `mcpServers`: (nome, caminho a partir do diretório do usuário).
const JSON_CLIENTS: [(&str, &[&str]); 1] =
    [("Antigravity", &[".gemini", "config", "mcp_config.json"])];

fn client_file(rel: &[&str]) -> Res<PathBuf> {
    Ok(rel.iter().fold(user_dir()?, |p, part| p.join(part)))
}

fn read_json(path: &Path) -> Res<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let text = fs::read_to_string(path)?;
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    Ok(serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?)
}

/// Escrita atômica: grava em arquivo temporário e substitui o original.
fn write_json(path: &Path, v: &Value) -> Res<()> {
    let tmp = PathBuf::from(format!("{}.fenda-tmp", path.display()));
    fs::write(&tmp, serde_json::to_string_pretty(v)? + "\n")?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Registra o servidor no arquivo do cliente; cliente sem o diretório de configuração é ignorado.
fn register_json(name: &str, rel: &[&str], exe: &str) -> String {
    let path = match client_file(rel) {
        Ok(p) => p,
        Err(e) => return format!("{name}: {e}"),
    };
    if !path.parent().is_some_and(Path::is_dir) {
        return format!("{name}: not found; skipped");
    }
    let r = (|| -> Res<()> {
        let mut v = read_json(&path)?;
        let root = v.as_object_mut().ok_or("not a JSON object")?;
        let servers = root.entry("mcpServers").or_insert_with(|| json!({}));
        let servers = servers
            .as_object_mut()
            .ok_or("mcpServers is not a JSON object")?;
        servers.insert("fenda".into(), json!({"command": exe, "args": ["mcp"]}));
        write_json(&path, &v)
    })();
    match r {
        Ok(()) => format!("{name}: registered in {}", path.display()),
        Err(e) => format!("{name}: could not register in {}: {e}", path.display()),
    }
}

pub fn install() -> Res<()> {
    let root = home()?;
    let bin = root.join("bin");
    fs::create_dir_all(&bin)?;

    let me = std::env::current_exe()?;
    let exe = bin.join(me.file_name().ok_or("no executable name")?);
    let same = |a: &Path, b: &Path| a.canonicalize().ok() == b.canonicalize().ok();
    if !same(&me, &exe) {
        fs::copy(&me, &exe).map_err(|e| {
            format!(
                "could not copy to {}: {e}. Close the AI clients that use fenda and run again.",
                exe.display()
            )
        })?;
    }
    eprintln!("program: {}", exe.display());

    let total: u64 = FILES.iter().map(|f| f.2).sum();
    eprintln!("models ({}):", mb(total));
    for (folder, file, size, sha) in FILES {
        let dir = root.join("models").join(folder);
        fs::create_dir_all(&dir)?;
        let dest = dir.join(file);
        let ok = dest.metadata().is_ok_and(|m| m.len() == size) && sha256(&dest)? == sha;
        if ok {
            eprintln!("  {folder}/{file}: already there");
            continue;
        }
        eprint!("  {folder}/{file} ({}) ", mb(size));
        download(
            &format!("{RELEASES}/{}/{file}", model_tag(folder)),
            &dest,
            size,
            sha,
        )?;
        eprintln!(" ok");
    }

    let exe_s = exe.to_string_lossy();
    for (name, bin, scope) in CLIS {
        eprintln!("{}", register_cli(name, bin, scope, &exe_s));
    }
    for (name, rel) in JSON_CLIENTS {
        eprintln!("{}", register_json(name, rel, &exe_s));
    }

    eprintln!(
        "\nDone. Put your .fnd books in {} (every project) or in a project's .fenda folder (that project only). Restart the AI client to see them.",
        root.display()
    );
    Ok(())
}

pub fn uninstall() -> Res<()> {
    let root = home()?;
    for (name, bin, scope) in CLIS {
        let removed = run_cli(bin, &[&["mcp", "remove", "fenda"][..], scope].concat())
            .is_some_and(|o| o.status.success());
        if removed {
            eprintln!("{name}: unregistered");
        }
    }
    for (name, rel) in JSON_CLIENTS {
        let Ok(path) = client_file(rel) else { continue };
        if !path.exists() {
            continue;
        }
        let mut v = read_json(&path)?;
        let removed = v
            .get_mut("mcpServers")
            .and_then(Value::as_object_mut)
            .and_then(|s| s.remove("fenda"))
            .is_some();
        if removed {
            write_json(&path, &v)?;
            eprintln!("{name}: unregistered");
        }
    }
    for (folder, file, _, _) in FILES {
        let _ = fs::remove_file(root.join("models").join(folder).join(file));
        let _ = fs::remove_dir(root.join("models").join(folder));
    }
    let _ = fs::remove_dir(root.join("models"));
    let exe = root
        .join("bin")
        .join(format!("fenda{}", std::env::consts::EXE_SUFFIX));
    if exe.exists() && fs::remove_file(&exe).is_err() {
        eprintln!(
            "delete {} by hand (it is the program running now)",
            exe.display()
        );
    }
    let _ = fs::remove_dir(root.join("bin"));
    let books = fs::read_dir(&root)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "fnd"))
                .count()
        })
        .unwrap_or(0);
    if books > 0 {
        eprintln!("kept {books} book file(s) in {}", root.display());
    }
    Ok(())
}
