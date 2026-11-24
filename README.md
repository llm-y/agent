# agent

Agentic AI CLI berbasiskan Rust. Nama perintahnya `llmy`: agent untuk terminal yang ringan,
memakai model lokal lewat [Ollama](https://ollama.com), Claude, atau API kompatibel OpenAI.

Memori cukup di `AGENTS.md`: bagian `## Memory` milik proyek, atau `~/.config/llmy/AGENTS.md`
yang terbawa ke semua proyek. Tanpa database.

Instalasi dan manual: <https://ll.my.id>

```sh
curl -fsSL https://ll.my.id/install.sh | sh   # Linux, macOS
irm https://ll.my.id/install.ps1 | iex        # Windows PowerShell
```

Dari sumber (Rust 1.91+ dan compiler C):

```sh
cargo install --locked --git https://github.com/llm-y/agent
```

## Pengembangan

```sh
cargo clippy --all-targets && cargo test && cargo build --release
```

Push ke `main` yang mengubah kode otomatis diuji, diberi tag, dan dirilis untuk Linux, macOS,
dan Windows (`.github/workflows/release.yml`). Pesan commit menentukan versinya: `feat:` atau
`#minor` menaikkan minor, `#major` menaikkan major, selain itu patch.

Lisensi GPL-3.0.
