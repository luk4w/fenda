# fenda

Reader for `.fnd` books: compiled PDFs with a table of contents, page-cited excerpts and vector search. It runs as an MCP server, so AI clients (Claude Code, Codex, Antigravity) can search and read the books.

## Install

```
fenda install
```

This copies the program to `~/.fenda/bin`, downloads the models (~690 MB, verified by SHA-256) and registers the MCP server in Claude Code, Codex and Antigravity. `fenda uninstall` reverts it and keeps the books.

## Books

- `~/.fenda/*.fnd`: available in every project.
- `<project>/.fenda/*.fnd`: available only in that project; it takes precedence over a global book with the same name.

Books are picked up on the next tool call; no restart is needed.

## MCP tools

| Tool | Purpose |
|---|---|
| `search` | find where a topic is covered |
| `outline` | chapters, or the subsections of one section |
| `open` | full text of a section |
| `restructure` | fix an entry of the table of contents |
| `remember` | save a note in the project memory |

## Command line

```
fenda search book.fnd "query"
fenda ask book.fnd "question"     # needs an OpenAI-compatible chat server
```

## Build

```
cargo build --release
```

Models run on the CPU through ONNX Runtime.
