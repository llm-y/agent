# agent

Agentic AI CLI berbasiskan Rust. Nama perintahnya `llmy`: agent untuk terminal yang ringan,
memakai model lokal lewat [Ollama](https://ollama.com), Claude, Google Gemini, atau API kompatibel OpenAI.

Simpan dan pilih API dari beberapa provider lewat `llmy api set` (disimpan di `~/.config/llmy/api.json`,
hanya terbaca pemilik) atau `/api` di dalam sesi. Memori cukup di `AGENTS.md`: bagian `## Memory`
milik proyek, atau `~/.config/llmy/AGENTS.md` yang terbawa ke semua proyek. Tanpa database.

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

### Penting: Gunakan GitHub API, bukan crawl HTML

Agen **harus** menggunakan output JSON dari API resmi GitHub (search, releases, contributors, contents, pulls, issues, stargazers),
bukan hasil crawl HTML. Meng-crawl halaman HTML GitHub boros token dan tidak reliable.

Jika agen sudah menemukan URL repositori lewat GitHub API atau Exa:
- **JANGAN** fetch HTML dari URL tersebut
- Gunakan output JSON yang sudah didapat dari API
- Semua tools GitHub hanya mengembalikan JSON dari API, bukan HTML crawling

## Tools HTTP & Crawling Tersedia:

### curl
Request HTTP dasar untuk download raw content dengan response header lengkap. Digunakan untuk:
- Download file dari URL apa saja
- Fetch metadata dari API REST tanpa library khusus
- Inspect headers dan status code

### Firecrawl/FastCRM
Crawling dan scraping terstruktur dengan ekstraksi konten:
- Endpoint: `GET /v1/crawl`, `GET /v1/scrape`
- Response: HTML bersih atau markdown yang diextract
- Kelebihan: Ekstraksi konten terstruktur, parsing otomatis

### Browserbase
Interactive browser automation untuk:
- Login ke situs modern dengan JavaScript
- Interaksi elemen DOM
- Screenshot dan video recording

Push ke `main` yang mengubah kode otomatis diuji, diberi tag, dan dirilis untuk Linux, macOS,
dan Windows (`.github/workflows/release.yml`). Pesan commit menentukan versinya: `feat:` atau
`#minor` menaikkan minor, `#major` menaikkan major, selain itu patch.

Lisensi GPL-3.0.
