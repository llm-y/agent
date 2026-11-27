//! `llmy setup`: TUI inline untuk memasang provider LLM gratis (protokol OpenAI). Kunci
//! diperiksa dengan mengambil daftar model, lalu profil disimpan seperti `llmy api set`.

use crate::api::{self, Config, Profile};
use crate::cancel::Cancel;
use crate::http::{self, Req};
use crate::provider::Kind;
use crate::ui::sanitize;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::{execute, terminal};
use std::io::{IsTerminal, Write};

const RST: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
/// Baris daftar yang tampil sekaligus.
const WINDOW: usize = 10;

pub struct Free {
    /// Nama profil di api.json, dipakai sebagai `-m <name>/<model>`.
    pub name: &'static str,
    /// Protokol: OpenAI, kecuali Gemini (native).
    pub kind: Kind,
    /// Env kunci, yang pertama untuk pesan; dipakai bila api.json tidak menyimpan kunci.
    pub env: &'static [&'static str],
    /// Bisa dipakai tanpa akun: tanpa kunci, header Authorization tidak dikirim.
    pub keyless: bool,
    pub label: &'static str,
    /// Alamat protokol OpenAI; `{account}` diganti Account ID (Cloudflare).
    pub base: &'static str,
    /// Alamat daftar model bila bukan `{base}/models`.
    pub list: Option<&'static str>,
    /// Alamat untuk memeriksa kunci bila daftar model bisa dibaca tanpa kunci.
    pub check: Option<&'static str>,
    /// Daftar model terbuka tanpa kunci, dan tidak ada `check`: kunci baru teruji saat dipakai.
    pub public_list: bool,
    pub key_url: &'static str,
    pub key_hint: &'static str,
    /// Disarankan untuk kerja agent (mendukung tool call), terbaik dulu.
    pub models: &'static [&'static str],
    /// Hanya model berakhiran `:free` (OpenRouter).
    pub only_free: bool,
    pub note: &'static str,
}

// Diperiksa dengan dokumentasi resmi pada 2026-11-26; ID model cepat berganti, jadi daftar
// dari server didahulukan dan yang di sini hanya saran.
pub const FREE: [Free; 10] = [
    Free {
        name: "groq", kind: Kind::OpenAi, env: &["GROQ_API_KEY"], keyless: false,
        label: "Groq", base: "https://api.groq.com/openai/v1", list: None, check: None, public_list: false,
        key_url: "https://console.groq.com/keys", key_hint: "",
        models: &["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"],
        only_free: false, note: "sangat cepat; gratis 30 permintaan dan 8K token per menit",
    },
    Free {
        name: "cerebras", kind: Kind::OpenAi, env: &["CEREBRAS_API_KEY"], keyless: false,
        label: "Cerebras", base: "https://api.cerebras.ai/v1", list: None, check: None, public_list: false,
        key_url: "https://cloud.cerebras.ai", key_hint: "",
        models: &["qwen-3.8-27b", "gpt-oss-120b"],
        only_free: false, note: "sangat cepat; kredit uji coba, gpt-oss-120b hanya 5 permintaan per menit",
    },
    Free {
        name: "openrouter", kind: Kind::OpenAi, env: &["OPENROUTER_API_KEY"], keyless: false,
        label: "OpenRouter (model :free)", base: "https://openrouter.ai/api/v1", list: None,
        check: Some("https://openrouter.ai/api/v1/key"), public_list: true,
        key_url: "https://openrouter.ai/keys", key_hint: "sk-or-v1-",
        models: &["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"],
        only_free: true, note: "model :free berganti-ganti; 20 permintaan per menit, 50 per hari",
    },
    Free {
        name: "cohere", kind: Kind::OpenAi, env: &["COHERE_API_KEY"], keyless: false,
        label: "Cohere", base: "https://api.cohere.ai/compatibility/v1",
        list: Some("https://api.cohere.com/v1/models?endpoint=chat"), check: None, public_list: false,
        key_url: "https://dashboard.cohere.com/api-keys", key_hint: "",
        models: &["command-a-plus-05-2026", "command-a-03-2025", "command-a-reasoning-08-2025"],
        only_free: false, note: "kunci trial: 20 permintaan per menit, 1.000 per bulan",
    },
    Free {
        name: "nvidia", kind: Kind::OpenAi, env: &["NVIDIA_API_KEY"], keyless: false,
        label: "NVIDIA NIM", base: "https://integrate.api.nvidia.com/v1", list: None, check: None, public_list: true,
        key_url: "https://build.nvidia.com (Get API Key di halaman model)", key_hint: "",
        models: &["nvidia/nemotron-3-ultra-550b-a55b", "z-ai/glm-5.3-flash", "moonshotai/kimi-k3.1-pro"],
        only_free: false, note: "gratis untuk pengembangan; sekitar 40 permintaan per menit",
    },
    Free {
        name: "cloudflare", kind: Kind::OpenAi, env: &["CLOUDFLARE_API_TOKEN"], keyless: false,
        label: "Cloudflare Workers AI", base: "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
        list: Some("https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search?task=Text%20Generation"),
        check: None, public_list: false,
        key_url: "https://dash.cloudflare.com/profile/api-tokens (API token, izin Workers AI)", key_hint: "",
        models: &["@cf/qwen/qwen3.8-27b", "@cf/zai-org/glm-4.7-flash", "@cf/openai/gpt-oss-120b"],
        only_free: false, note: "10.000 neuron per hari gratis; butuh Account ID",
    },
    // Ditambahkan 2026-11-26 dari dokumentasi resmi dan uji endpoint (tanpa kunci / kunci salah).
    Free {
        name: "zai", kind: Kind::OpenAi, env: &["ZAI_API_KEY", "Z_AI_API_KEY"], keyless: false,
        label: "Z.ai (GLM Flash)", base: "https://api.z.ai/api/paas/v4", list: None, check: None, public_list: false,
        key_url: "https://z.ai/manage-apikey/apikey-list (Standard API Key, bukan Coding Plan)", key_hint: "",
        models: &["glm-4.7-flash", "glm-4.5-flash"],
        only_free: false, note: "gratis hanya glm-4.7-flash dan glm-4.5-flash; glm-5.3-flash dan *-flashx berbayar",
    },
    Free {
        name: "bigmodel", kind: Kind::OpenAi, env: &["ZHIPU_API_KEY", "BIGMODEL_API_KEY"], keyless: false,
        label: "Zhipu BigModel (Tiongkok)", base: "https://open.bigmodel.cn/api/paas/v4", list: None, check: None, public_list: false,
        key_url: "https://bigmodel.cn/usercenter/proj-mgmt/apikeys", key_hint: "",
        models: &["glm-4.7-flash", "glm-4.5-flash", "glm-4-flash-250414"],
        only_free: false, note: "GLM Flash gratis; batas paralel menurut level akun",
    },
    Free {
        name: "gemini", kind: Kind::Gemini, env: &["GEMINI_API_KEY", "GOOGLE_API_KEY"], keyless: false,
        label: "Google AI Studio (Gemini)", base: "https://generativelanguage.googleapis.com/v1beta", list: None, check: None, public_list: false,
        key_url: "https://aistudio.google.com/apikey", key_hint: "AIza",
        models: &["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"],
        only_free: false, note: "batas harian kecil (Flash sekitar 20 per hari); data tier gratis dipakai melatih model Google",
    },
    Free {
        name: "kilo", kind: Kind::OpenAi, env: &["KILO_API_KEY"], keyless: true,
        label: "Kilo Gateway (tanpa akun)", base: "https://ai-gateway.kilo.ai/api/v1", list: None,
        check: Some("https://api.kilo.ai/api/profile"), public_list: true,
        key_url: "https://app.kilo.ai (opsional; tanpa kunci pun jalan)", key_hint: "",
        models: &["nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3.5-lightning:free", "kilo-auto/free"],
        only_free: true, note: "model :free tanpa akun, 200 permintaan per jam per IP; sebagian melatih dari prompt",
    },
];

pub fn find(name: &str) -> Option<&'static Free> {
    FREE.iter().find(|f| f.name == name)
}

// ---- tampilan ----

struct Screen {
    out: std::io::Stderr,
    drawn: usize,
}

impl Screen {
    fn width() -> usize {
        // `s` bisa melaporkan 0x0 di awal; lihat `ui::size`.
        crate::ui::size(terminal::size().unwrap_or((0, 0))).0.min(200)
    }

    /// Gambar ulang area yang sama: naik ke baris pertama, hapus, cetak.
    fn draw(&mut self, lines: &[(&str, String)]) {
        let w = Self::width() - 1;
        let mut s = String::new();
        if self.drawn > 0 {
            s += &format!("\r\x1b[{}A", self.drawn);
        }
        s += "\r\x1b[J";
        for (style, l) in lines {
            let t: String = sanitize(l).replace('\n', " ").chars().take(w).collect();
            s += &format!("{style}{t}{RST}\r\n");
        }
        self.drawn = lines.len();
        let _ = self.out.write_all(s.as_bytes());
        let _ = self.out.flush();
    }

    /// Biarkan isi terakhir tetap di layar.
    fn keep(&mut self) {
        self.drawn = 0;
    }
}

enum Key {
    Char(char),
    Paste(String),
    Back,
    Up,
    Down,
    PageUp,
    PageDown,
    Enter,
    Esc,
    Redraw,
}

fn key() -> Result<Key, String> {
    loop {
        match event::read().map_err(|e| e.to_string())? {
            Event::Key(k) if k.kind != KeyEventKind::Release => {
                let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                return Ok(match k.code {
                    KeyCode::Char('c') | KeyCode::Char('d') if ctrl => return Err("dibatalkan".into()),
                    KeyCode::Char('u') if ctrl => Key::Paste("\u{15}".into()),
                    KeyCode::Char(c) if !ctrl => Key::Char(c),
                    KeyCode::Backspace => Key::Back,
                    KeyCode::Up => Key::Up,
                    KeyCode::Down => Key::Down,
                    KeyCode::PageUp => Key::PageUp,
                    KeyCode::PageDown => Key::PageDown,
                    KeyCode::Enter => Key::Enter,
                    KeyCode::Esc => Key::Esc,
                    _ => continue,
                });
            }
            Event::Paste(s) => return Ok(Key::Paste(s)),
            Event::Resize(..) => return Ok(Key::Redraw),
            _ => {}
        }
    }
}

pub struct Item {
    pub label: String,
    pub note: String,
}

/// Pilih satu dari daftar. `filter`: ketikan menyaring daftar. None = Esc.
fn pick(scr: &mut Screen, title: &str, help: &str, items: &[Item], filter: bool) -> Result<Option<usize>, String> {
    let (mut sel, mut top, mut q) = (0usize, 0usize, String::new());
    loop {
        let shown: Vec<usize> = (0..items.len())
            .filter(|&i| q.is_empty() || items[i].label.to_lowercase().contains(&q.to_lowercase()))
            .collect();
        sel = sel.min(shown.len().saturating_sub(1));
        if sel < top {
            top = sel;
        }
        if sel >= top + WINDOW {
            top = sel + 1 - WINDOW;
        }
        let mut lines: Vec<(&str, String)> = vec![(BOLD, title.to_string())];
        if filter {
            lines.push((CYAN, format!("  cari: {q}▏")));
        }
        for (n, &i) in shown.iter().enumerate().skip(top).take(WINDOW) {
            let it = &items[i];
            let mark = if n == sel { "›" } else { " " };
            let num = match (filter, n < 9) {
                (false, true) => format!("{}. ", n + 1),
                (false, false) => "   ".into(),
                (true, _) => String::new(),
            };
            let note = if it.note.is_empty() { String::new() } else { format!("  {}", it.note) };
            lines.push((if n == sel { BOLD } else { "" }, format!("{mark} {num}{}{note}", it.label)));
        }
        if shown.is_empty() {
            lines.push((DIM, "  (tidak ada yang cocok)".into()));
        } else if shown.len() > WINDOW {
            lines.push((DIM, format!("  {}–{} dari {}", top + 1, (top + WINDOW).min(shown.len()), shown.len())));
        }
        lines.push((DIM, help.to_string()));
        scr.draw(&lines);
        match key()? {
            Key::Up => sel = sel.saturating_sub(1),
            Key::Down => sel = (sel + 1).min(shown.len().saturating_sub(1)),
            Key::PageUp => sel = sel.saturating_sub(WINDOW),
            Key::PageDown => sel = (sel + WINDOW).min(shown.len().saturating_sub(1)),
            Key::Enter if !shown.is_empty() => return Ok(Some(shown[sel])),
            Key::Esc => return Ok(None),
            Key::Char(c @ '1'..='9') if !filter => {
                let n = c as usize - '1' as usize;
                if n < shown.len() {
                    return Ok(Some(shown[n]));
                }
            }
            Key::Char(c) if filter => {
                q.push(c);
                sel = 0;
            }
            Key::Back if filter => {
                q.pop();
                sel = 0;
            }
            Key::Paste(s) if filter && s != "\u{15}" => {
                q += s.trim();
                sel = 0;
            }
            _ => {}
        }
    }
}

/// Satu baris masukan. `secret`: tampil sebagai titik. None = Esc.
fn input(scr: &mut Screen, head: &[(&str, String)], prompt: &str, secret: bool) -> Result<Option<String>, String> {
    let mut buf = String::new();
    loop {
        let shown = if secret { "•".repeat(buf.chars().count().min(40)) } else { buf.clone() };
        let mut lines = head.to_vec();
        lines.push(("", format!("{prompt}{shown}▏")));
        lines.push((DIM, "Enter simpan · Esc kembali · Ctrl+U hapus".into()));
        scr.draw(&lines);
        match key()? {
            Key::Enter => return Ok(Some(buf.trim().to_string())),
            Key::Esc => return Ok(None),
            Key::Back => {
                buf.pop();
            }
            Key::Paste(s) if s == "\u{15}" => buf.clear(),
            Key::Paste(s) => buf += s.trim(),
            Key::Char(c) => buf.push(c),
            _ => {}
        }
    }
}

// ---- jaringan ----

fn get(url: &str, key: &str, kind: Kind, cancel: &Cancel) -> Result<serde_json::Value, http::Fail> {
    let (root, path) = match url.find("://").and_then(|i| url[i + 3..].find('/').map(|j| i + 3 + j)) {
        Some(i) => (&url[..i], url[i..].to_string()),
        None => (url, "/".to_string()),
    };
    let mut headers = vec![("Accept", "application/json".to_string())];
    match kind {
        _ if key.is_empty() => {}
        Kind::Gemini => headers.push(("x-goog-api-key", key.to_string())),
        _ => headers.push(("Authorization", format!("Bearer {key}"))),
    }
    let mut raw = String::new();
    http::request(&Req { base: root, method: "GET", path: &path, headers: &headers, body: "" }, cancel, &mut |l| {
        raw.push_str(l);
        Ok(())
    })?;
    serde_json::from_str(&raw).map_err(|e| http::Fail::from(format!("balasan bukan JSON: {e}")))
}

/// ID model yang bisa dipakai agent. Bentuk OpenAI `data[].id`; Cohere `models[].name`;
/// Cloudflare `result[].name`. Model yang menyatakan tidak mendukung tool dibuang.
pub fn ids(v: &serde_json::Value) -> Vec<String> {
    let list = v["data"].as_array().or(v["models"].as_array()).or(v["result"].as_array()).cloned().unwrap_or_default();
    let mut out: Vec<String> = list
        .iter()
        .filter(|m| {
            let no_tools = m["supported_parameters"].as_array().is_some_and(|p| !p.iter().any(|x| x == "tools"))
                || m["capabilities"]["function_calling"] == false
                || m["tools_calling"] == false
                || m["is_deprecated"] == true
                // Gemini: model embedding dan sejenisnya tidak bisa chat.
                || m["supportedGenerationMethods"].as_array().is_some_and(|p| !p.iter().any(|x| x == "generateContent"));
            !no_tools
        })
        .filter_map(|m| m["id"].as_str().or(m["name"].as_str()).map(|s| s.strip_prefix("models/").unwrap_or(s).to_string()))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Periksa kunci (bila ada alamatnya) lalu ambil daftar model.
fn models(f: &Free, base: &str, account: &str, key: &str, cancel: &Cancel) -> Result<Vec<String>, http::Fail> {
    let test = std::env::var("LLMY_SETUP_BASE").is_ok();
    if let Some(c) = f.check.filter(|_| !test && !key.is_empty()) {
        get(c, key, f.kind, cancel)?;
    }
    let url = match f.list.filter(|_| !test) {
        Some(l) => l.replace("{account}", account),
        None if f.kind == Kind::Gemini => format!("{base}/models?pageSize=1000"),
        None => format!("{base}/models"),
    };
    match get(&url, key, f.kind, cancel) {
        Ok(v) => Ok(ids(&v)),
        // Gemini membalas kunci salah dengan 400, bukan 401.
        Err(mut e) if e.status == 400 && (e.msg.contains("API_KEY_INVALID") || e.msg.contains("API key not valid")) => {
            e.status = 401;
            Err(e)
        }
        Err(e) => Err(e),
    }
}

/// Urutan tampil: yang disarankan dan tersedia dulu, lalu sisanya.
pub fn order(f: &Free, fetched: &[String]) -> Vec<Item> {
    let ok = |m: &str| !f.only_free || m.ends_with(":free") || m.ends_with("/free");
    let mut v: Vec<Item> = vec![];
    let have = |m: &str| fetched.is_empty() || fetched.iter().any(|x| x == m);
    for m in f.models.iter().filter(|m| have(m)) {
        v.push(Item { label: m.to_string(), note: "★ disarankan".into() });
    }
    for m in fetched.iter().filter(|m| ok(m) && !f.models.contains(&m.as_str())) {
        v.push(Item { label: m.clone(), note: String::new() });
    }
    v
}

fn base_of(f: &Free, account: &str) -> String {
    if let Ok(b) = std::env::var("LLMY_SETUP_BASE") {
        // Untuk uji: semua provider diarahkan ke server tiruan.
        return b.trim_end_matches('/').to_string();
    }
    f.base.replace("{account}", account)
}

pub fn valid_account(s: &str) -> bool {
    s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---- alur ----

pub fn cli() -> i32 {
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        eprintln!("llmy setup butuh terminal. Tanpa terminal pakai: llmy api set <nama> --key ... --model ...");
        return 2;
    }
    if terminal::enable_raw_mode().is_err() {
        eprintln!("llmy setup: terminal tidak mendukung mode raw");
        return 2;
    }
    let _ = execute!(std::io::stderr(), event::EnableBracketedPaste);
    let mut scr = Screen { out: std::io::stderr(), drawn: 0 };
    let r = run(&mut scr);
    let _ = execute!(std::io::stderr(), event::DisableBracketedPaste);
    let _ = terminal::disable_raw_mode();
    match r {
        Ok(Some(msg)) => {
            eprintln!("{msg}");
            0
        }
        Ok(None) => {
            eprintln!("Dibatalkan; tidak ada yang disimpan.");
            1
        }
        Err(e) => {
            eprintln!("\r\nllmy setup: {e}");
            1
        }
    }
}

fn run(scr: &mut Screen) -> Result<Option<String>, String> {
    let cfg = Config::load_strict()?;
    let items: Vec<Item> = FREE
        .iter()
        .map(|f| {
            // Yang tersimpan didahulukan: catatan boleh terpotong di layar sempit.
            let saved = cfg.providers.get(f.name).map(|p| format!("[tersimpan{}] ", p.model.as_deref().map(|m| format!(": {m}")).unwrap_or_default())).unwrap_or_default();
            Item { label: format!("{:<26}", f.label), note: format!("{saved}{}", f.note) }
        })
        .collect();
    loop {
        let Some(i) = pick(scr, "Pasang provider LLM gratis", "↑↓ atau 1-9 pilih · Enter lanjut · Esc keluar", &items, false)? else {
            return Ok(None);
        };
        if let Some(msg) = provider(scr, &FREE[i])? {
            return Ok(Some(msg));
        }
    }
}

/// Satu provider. None = kembali ke daftar provider.
fn provider(scr: &mut Screen, f: &Free) -> Result<Option<String>, String> {
    let mut cfg = Config::load_strict()?;
    let old = cfg.providers.get(f.name).cloned();
    let mut head: Vec<(&str, String)> = vec![
        (BOLD, f.label.to_string()),
        ("", format!("  Buat kunci di: {}", f.key_url)),
        (DIM, format!("  {}", f.note)),
    ];
    let mut account = String::new();
    if f.base.contains("{account}") {
        let had = old.as_ref().and_then(|p| p.url.as_deref()).and_then(|u| u.split("/accounts/").nth(1)).and_then(|r| r.split('/').next()).unwrap_or("").to_string();
        loop {
            let prompt = if had.is_empty() { "Account ID: ".to_string() } else { format!("Account ID (Enter = {had}): ") };
            let mut h = head.clone();
            h.push((DIM, "  Account ID: di dashboard tekan Ctrl/Cmd+K lalu \"Copy account ID\", atau Workers & Pages → Account Details.".into()));
            let Some(a) = input(scr, &h, &prompt, false)? else { return Ok(None) };
            let a = if a.is_empty() { had.clone() } else { a };
            if valid_account(&a) {
                account = a;
                break;
            }
            head.push((RED, "  Account ID berupa 32 karakter heksadesimal.".into()));
        }
        head.retain(|(s, _)| *s != RED);
    }
    let base = base_of(f, &account);
    let cancel = Cancel::default();
    let fetched = loop {
        let keep = old.as_ref().and_then(|p| p.key.clone());
        let from_env = f.env.iter().find(|e| crate::provider::env(e).is_some());
        let prompt = match (&keep, from_env) {
            (Some(k), _) => format!("Kunci API (Enter = pakai {}): ", api::mask(k)),
            (None, Some(e)) => format!("Kunci API (Enter = pakai env {e}): "),
            (None, None) if f.keyless => "Kunci API (Enter = tanpa kunci): ".into(),
            (None, None) if f.key_hint.is_empty() => "Kunci API: ".into(),
            (None, None) => format!("Kunci API ({}...): ", f.key_hint),
        };
        let Some(k) = input(scr, &head, &prompt, true)? else { return Ok(None) };
        let key = match (k.is_empty(), keep, from_env) {
            (false, _, _) => k,
            (true, Some(k), _) => k,
            (true, None, Some(e)) => crate::provider::env(e).unwrap_or_default(),
            (true, None, None) if f.keyless => String::new(),
            (true, None, None) => continue,
        };
        let mut h = head.clone();
        h.push((DIM, "  Memeriksa kunci dan mengambil daftar model…".into()));
        scr.draw(&h);
        match models(f, &base, &account, &key, &cancel) {
            Ok(m) => break (key, m),
            Err(e) if matches!(e.status, 401 | 403) => {
                head.retain(|(s, _)| *s != RED);
                head.push((RED, format!("  Kunci ditolak ({}). Periksa lalu ulangi.", e.status)));
            }
            // Daftar model tidak tersedia, tetapi kuncinya belum tentu salah: lanjut dengan yang disarankan.
            Err(e) => {
                head.retain(|(s, _)| *s != RED);
                head.push((RED, format!("  Daftar model tidak terambil: {}", e.msg.chars().take(120).collect::<String>())));
                break (key, vec![]);
            }
        }
    };
    let (key, list) = fetched;
    let mut items = order(f, &list);
    items.push(Item { label: "Ketik nama model lain…".into(), note: String::new() });
    let title = if list.is_empty() {
        format!("{} · pilih model (daftar saran llmy; kunci belum diperiksa)", f.label)
    } else if key.is_empty() {
        format!("{} · {} model · tanpa kunci", f.label, items.len() - 1)
    } else if f.public_list && f.check.is_none() {
        format!("{} · {} model · kunci baru diperiksa saat dipakai", f.label, items.len() - 1)
    } else {
        format!("{} · kunci diterima · {} model", f.label, items.len() - 1)
    };
    let model = loop {
        let Some(i) = pick(scr, &title, "ketik untuk menyaring · ↑↓ pilih · Enter lanjut · Esc kembali", &items, true)? else {
            return Ok(None);
        };
        if i + 1 < items.len() {
            break items[i].label.clone();
        }
        if let Some(m) = input(scr, &[(BOLD, title.clone())], "Nama model: ", false)?.filter(|m| !m.is_empty()) {
            break m;
        }
    };
    let yes = [Item { label: "Ya".into(), note: format!("`llmy` langsung memakai {}/{model}", f.name) }, Item { label: "Tidak".into(), note: format!("pakai dengan `llmy -m {}/{model}`", f.name) }];
    let Some(d) = pick(scr, "Jadikan provider bawaan?", "1/2 atau ↑↓ · Enter · Esc kembali", &yes, false)? else {
        return Ok(None);
    };
    cfg = Config::load_strict()?;
    // Kunci dari env tidak disalin ke api.json: llmy membacanya lagi dari env.
    let from_env = !key.is_empty() && f.env.iter().any(|e| crate::provider::env(e).as_deref() == Some(key.as_str()));
    let key = (!key.is_empty() && !from_env).then_some(key);
    cfg.providers.insert(f.name.into(), Profile { protocol: f.kind, url: Some(base.clone()), key, model: Some(model.clone()) });
    if d == 0 {
        cfg.default = Some(f.name.into());
    }
    cfg.save()?;
    scr.draw(&[(GREEN, format!("✓ {} tersimpan: model {model}", f.label))]);
    scr.keep();
    let run = if d == 0 { "llmy".to_string() } else { format!("llmy -m {}/{model}", f.name) };
    let file = api::file().map(|p| p.display().to_string()).unwrap_or_default();
    Ok(Some(format!("  Jalankan: {run}\n  Disimpan di {file} (hanya terbaca pemiliknya).")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urutan_model() {
        let f = find("openrouter").unwrap();
        let got = vec!["a/b".to_string(), "x/y:free".into(), f.models[0].to_string()];
        let v: Vec<String> = order(f, &got).into_iter().map(|i| i.label).collect();
        // Disarankan yang tersedia dulu, lalu yang :free saja.
        assert_eq!(v, vec![f.models[0].to_string(), "x/y:free".into()]);
        let g = find("groq").unwrap();
        assert_eq!(order(g, &[]).len(), g.models.len());
    }

    #[test]
    fn model_tanpa_tool_dibuang() {
        let v = serde_json::json!({"data": [
            {"id": "a:free", "supported_parameters": ["tools", "temperature"]},
            {"id": "b:free", "supported_parameters": ["temperature"]},
            {"id": "c", "capabilities": {"function_calling": false}},
            {"id": "d"}]});
        assert_eq!(ids(&v), vec!["a:free", "d"]);
        let c = serde_json::json!({"models": [{"name": "command-a-03-2025"}, {"name": "old", "is_deprecated": true}]});
        assert_eq!(ids(&c), vec!["command-a-03-2025"]);
        let cf = serde_json::json!({"success": true, "result": [{"name": "@cf/qwen/qwen3.8-27b"}]});
        assert_eq!(ids(&cf), vec!["@cf/qwen/qwen3.8-27b"]);
    }

    #[test]
    fn account_cloudflare() {
        assert!(valid_account("0123456789abcdef0123456789ABCDEF"));
        assert!(!valid_account("123") && !valid_account("0123456789abcdef0123456789abcdeg"));
        let f = find("cloudflare").unwrap();
        assert_eq!(f.base.replace("{account}", "abc"), "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1");
    }

    #[test]
    fn nama_profil_sah() {
        for f in &FREE {
            assert!(api::valid_name(f.name), "{}", f.name);
            assert!(f.base.starts_with("https://") && !f.models.is_empty() && !f.env.is_empty());
        }
        let names: std::collections::HashSet<_> = FREE.iter().map(|f| f.name).collect();
        assert_eq!(names.len(), FREE.len());
    }
}
