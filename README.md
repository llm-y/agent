# agent

Agentic AI CLI berbasiskan Rust. Nama perintahnya `llmy`: agent untuk terminal yang ringan,
memakai model lokal lewat [Ollama](https://ollama.com), Claude, Google Gemini, atau API kompatibel OpenAI.

Simpan dan pilih API dari beberapa provider lewat `llmy api set` (disimpan di `~/.config/llmy/api.json`,
hanya terbaca pemilik) atau `/api` di dalam sesi. Memori cukup di `AGENTS.md`: bagian `## Memory`
milik proyek, atau `~/.config/llmy/AGENTS.md` yang terbawa ke semua proyek. Tanpa database.

Tool: `read` (juga per simbol), `head`, `tail`, `grep`, `edit` (juga beberapa bagian sekaligus), `write`,
`bash`, `task` (sub-agent dengan context baru), `essence`, `remember`, dan tool internet
`curl`, `github`, `openalex`, `doi`, `spider`; `tavily`, `exa`, `linkup`, `firecrawl`, `scopus`, dan
`brightdata` muncul bila kuncinya ada di env. Setiap tindakan ditanya, kecuali dengan `--allow` atau
mode auto (`--auto`, `/auto`, Shift+Tab).

Bahasa utama: Rust, JavaScript (croot.js), LaTeX, dan Python. Setelah edit, llmy sendiri menjalankan
pemeriksaan proyek (`cargo check`, `node --check`, kompilasi draft LaTeX, `py_compile`, juga `go build`, `tsc`,
atau `--verify <perintah>`) dan memberikan galatnya ke model. Isi file yang sudah gagal ditolak bila ditulis lagi.

Provider gratis (Groq, Cerebras, OpenRouter `:free`, Cohere, NVIDIA NIM, Cloudflare Workers AI,
Z.ai dan Zhipu BigModel (GLM Flash), Google AI Studio, Kilo, serta DeepSeek yang berbayar) bisa
dipasang lewat menu `llmy setup`, atau cukup lewat env kuncinya (`GROQ_API_KEY`, `ZAI_API_KEY`, ...).
`--fallback a,b` berpindah ke model lain saat kuota habis.

Alur bawaan: satu loop, satu model. `--mode bdi` membagi kerja: perencana (bisa model lain, `--plan`)
memecah tugas, tiap tujuan dikerjakan pekerja dengan context baru, llmy memverifikasi, dan file
dikembalikan bila buntu. `/undo` mengembalikan file yang diubah giliran terakhir, termasuk lewat `bash`.

Harness: saat model menyatakan selesai, llmy menjalankan uji proyek dan mengembalikan kegagalannya ke model;
galat pemeriksaan disertai potongan kode di lokasinya (16 tugas, qwen3.5:9b: 33/48 → 45/48 benar); peta proyek
opsional (`LLMY_MAP=on`); `edit` toleran indentasi;
argumen tool yang cacat diperbaiki; sub-agent baca-saja berjalan paralel; `llmy stats` meringkas metrik lokal.

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
