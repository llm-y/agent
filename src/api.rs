//! Pengaturan API untuk beberapa provider (disimpan di ~/.config/llmy/api.json,
//! izin 0600) beserta perintah CLI `llmy api` dan `/api`.

use crate::provider::Kind;
use crate::{anthropic, gemini, openai, sys};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub protocol: Kind,
    pub url: Option<String>,
    pub key: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub default: Option<String>,
    pub providers: BTreeMap<String, Profile>,
}

/// Lokasi berkas konfigurasi API (hanya dapat dibaca pemilik).
pub fn file() -> Option<PathBuf> {
    sys::config_dir().map(|d| d.join("api.json"))
}

/// Nama provider/alias yang dikenal beserta protokol dan alamat bawaannya.
pub fn preset(name: &str) -> Option<(Kind, &'static str)> {
    Some(match name {
        "ollama" => (Kind::Ollama, "http://localhost:11434"),
        "anthropic" => (Kind::Anthropic, anthropic::BASE),
        "openai" => (Kind::OpenAi, openai::BASE),
        "gemini" => (Kind::Gemini, gemini::BASE),
        "deepseek" => (Kind::OpenAi, "https://api.deepseek.com"),
        "openrouter" => (Kind::OpenAi, "https://openrouter.ai/api/v1"),
        "groq" => (Kind::OpenAi, "https://api.groq.com/openai/v1"),
        "mistral" => (Kind::OpenAi, "https://api.mistral.ai/v1"),
        "xai" => (Kind::OpenAi, "https://api.x.ai/v1"),
        _ => return None,
    })
}

pub fn is_builtin(name: &str) -> bool {
    matches!(name, "ollama" | "anthropic" | "openai" | "gemini")
}

pub fn parse_kind(s: &str) -> Option<Kind> {
    match s.to_ascii_lowercase().as_str() {
        "ollama" => Some(Kind::Ollama),
        "anthropic" | "claude" => Some(Kind::Anthropic),
        "openai" | "openai-compatible" => Some(Kind::OpenAi),
        "gemini" | "google" => Some(Kind::Gemini),
        _ => None,
    }
}

pub fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Ollama => "ollama",
        Kind::Anthropic => "anthropic",
        Kind::OpenAi => "openai",
        Kind::Gemini => "gemini",
    }
}

pub fn default_url(k: Kind) -> &'static str {
    match k {
        Kind::Ollama => "http://localhost:11434",
        Kind::Anthropic => anthropic::BASE,
        Kind::OpenAi => openai::BASE,
        Kind::Gemini => gemini::BASE,
    }
}

pub fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 40
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

pub fn mask(k: &str) -> String {
    let n = k.chars().count();
    if n <= 8 {
        return "••••".into();
    }
    let head: String = k.chars().take(4).collect();
    let tail: String = k.chars().skip(n - 4).collect();
    format!("{head}...{tail}")
}

fn non_empty(v: &Value) -> Option<String> {
    v.as_str().map(str::trim).filter(|s| !s.is_empty()).map(String::from)
}

impl Config {
    pub fn load() -> Self {
        Self::load_strict().unwrap_or_default()
    }

    pub fn load_strict() -> Result<Self, String> {
        let Some(path) = file() else {
            return Ok(Self::default());
        };
        let raw = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let v: Value = serde_json::from_slice(&raw)
            .map_err(|e| format!("{} rusak: {e}", path.display()))?;
        let mut cfg = Config {
            default: non_empty(&v["default"]),
            providers: BTreeMap::new(),
        };
        if let Some(map) = v["providers"].as_object() {
            for (name, p) in map {
                let Some(proto) = p["protocol"].as_str().and_then(parse_kind) else { continue };
                cfg.providers.insert(name.clone(), Profile {
                    protocol: proto,
                    url: non_empty(&p["url"]).map(|u| u.trim_end_matches('/').to_string()),
                    key: non_empty(&p["key"]),
                    model: non_empty(&p["model"]),
                });
            }
        }
        Ok(cfg)
    }

    pub fn save(&self) -> Result<(), String> {
        let path = file().ok_or("folder konfigurasi tidak ditemukan")?;
        if let Some(d) = path.parent() {
            sys::private_dir(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let mut providers = serde_json::Map::new();
        for (name, p) in &self.providers {
            let mut obj = serde_json::Map::new();
            obj.insert("protocol".into(), json!(kind_name(p.protocol)));
            if let Some(u) = &p.url {
                obj.insert("url".into(), json!(u));
            }
            if let Some(k) = &p.key {
                obj.insert("key".into(), json!(k));
            }
            if let Some(m) = &p.model {
                obj.insert("model".into(), json!(m));
            }
            providers.insert(name.clone(), Value::Object(obj));
        }
        let mut root = json!({ "providers": Value::Object(providers) });
        if let Some(d) = &self.default {
            root["default"] = json!(d);
        }
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let mut body = serde_json::to_string_pretty(&root).map_err(|e| e.to_string())?;
        body.push('\n');
        let io = |e: std::io::Error| format!("{}: {e}", path.display());
        sys::private_file(&tmp).and_then(|mut f| f.write_all(body.as_bytes())).map_err(io)?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            io(e)
        })
    }
}

#[derive(Default)]
pub struct SetOpts {
    pub protocol: Option<Kind>,
    pub url: Option<String>,
    pub key: Option<String>,
    pub no_key: bool,
    pub model: Option<String>,
    pub default: bool,
    pub session: bool,
}

/// Pecah baris perintah dengan dukungan tanda kutip `'...'` dan `"..."`.
pub fn split_words(s: &str) -> Result<Vec<String>, String> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut has = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) if c == q => quote = None,
            Some('\\') if quote == Some('"') => {
                cur.push(chars.next().unwrap_or('\\'));
                has = true;
            }
            Some(_) => {
                cur.push(c);
                has = true;
            }
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                has = true;
            }
            None if c.is_whitespace() => {
                if has {
                    out.push(std::mem::take(&mut cur));
                    has = false;
                }
            }
            None => {
                cur.push(c);
                has = true;
            }
        }
    }
    if quote.is_some() {
        return Err("tanda kutip tidak ditutup".into());
    }
    if has {
        out.push(cur);
    }
    Ok(out)
}

pub fn parse_set_opts(args: &[String]) -> Result<SetOpts, String> {
    let mut o = SetOpts::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |i: &mut usize, flag: &str| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("{flag} butuh nilai"))
        };
        if let Some(v) = a.strip_prefix("protocol=") {
            o.protocol = Some(parse_kind(v).ok_or_else(|| format!("protokol tidak dikenal: {v}"))?);
        } else if let Some(v) = a.strip_prefix("url=") {
            o.url = Some(v.into());
        } else if let Some(v) = a.strip_prefix("key=") {
            o.key = Some(v.into());
        } else if let Some(v) = a.strip_prefix("model=") {
            o.model = Some(v.into());
        } else {
            match a {
                "--protocol" | "-p" => {
                    let v = next(&mut i, a)?;
                    o.protocol = Some(parse_kind(&v).ok_or_else(|| format!("protokol tidak dikenal: {v}"))?);
                }
                "--url" | "-u" => o.url = Some(next(&mut i, a)?),
                "--key" | "-k" => o.key = Some(next(&mut i, a)?),
                "--no-key" => o.no_key = true,
                "--model" | "-m" => o.model = Some(next(&mut i, a)?),
                "--default" => o.default = true,
                "--session" => o.session = true,
                _ if a.starts_with('-') => return Err(format!("opsi tidak dikenal: {a}")),
                _ if a.contains("://") && !a.starts_with("http://") && !a.starts_with("https://") => {
                    return Err(format!("URL harus diawali http:// atau https:// ({a})"));
                }
                _ if (a.starts_with("http://") || a.starts_with("https://")) && o.url.is_none() => {
                    o.url = Some(a.into());
                }
                _ if o.key.is_none() => o.key = Some(a.into()),
                _ if o.model.is_none() => o.model = Some(a.into()),
                _ => return Err(format!("argumen berlebih: {a}")),
            }
        }
        i += 1;
    }
    if let Some(u) = &o.url {
        let u = u.trim();
        if !u.starts_with("http://") && !u.starts_with("https://") {
            return Err(format!("URL harus diawali http:// atau https:// ({u})"));
        }
    }
    Ok(o)
}

/// Bangun profil gabungan dari profil lama/preset + opsi baru.
pub fn build_profile(name: &str, existing: Option<&Profile>, o: SetOpts) -> Result<Profile, String> {
    if !valid_name(name) {
        return Err(format!("nama provider tidak valid: {name:?}"));
    }
    let pre = preset(name);
    let protocol = o
        .protocol
        .or_else(|| existing.map(|p| p.protocol))
        .or_else(|| pre.map(|(k, _)| k))
        .unwrap_or(Kind::OpenAi);
    let url = o
        .url
        .map(|u| u.trim().trim_end_matches('/').to_string())
        .or_else(|| existing.and_then(|p| p.url.clone()))
        .or_else(|| pre.map(|(_, u)| u.to_string()));
    let key = if o.no_key {
        None
    } else {
        o.key
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .or_else(|| existing.and_then(|p| p.key.clone()))
    };
    let model = o
        .model
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .or_else(|| existing.and_then(|p| p.model.clone()));
    Ok(Profile { protocol, url, key, model })
}

pub fn format_list(cfg: &Config, session: &BTreeMap<String, Profile>, active: Option<&str>) -> String {
    let mut out = String::new();
    let mut names: Vec<String> = cfg.providers.keys().cloned().collect();
    for k in session.keys() {
        if !cfg.providers.contains_key(k) {
            names.push(k.clone());
        }
    }
    if names.is_empty() {
        out.push_str("Belum ada provider tersimpan.\n");
    } else {
        out.push_str("Provider tersimpan:\n");
        for name in &names {
            let s = session.get(name);
            let d = cfg.providers.get(name);
            let p = s.or(d).unwrap();
            let url = p.url.as_deref().unwrap_or_else(|| default_url(p.protocol));
            let key = match &p.key {
                Some(k) if s.is_some() => format!("{} (sesi)", mask(k)),
                Some(k) => mask(k),
                None if p.protocol == Kind::Ollama => "-".into(),
                None => "(kosong)".into(),
            };
            let mut tags = vec![];
            if active == Some(name.as_str()) {
                tags.push("aktif");
            }
            if cfg.default.as_deref() == Some(name.as_str()) {
                tags.push("bawaan");
            }
            if s.is_some() && d.is_some() {
                tags.push("ditimpa sesi");
            } else if s.is_some() {
                tags.push("hanya sesi");
            }
            let tag = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(", ")) };
            let model = p.model.as_deref().map(|m| format!(" · model {m}")).unwrap_or_default();
            out.push_str(&format!("  {name}{tag}: {} · {url} · kunci {key}{model}\n", kind_name(p.protocol)));
        }
    }
    let mut envs = vec![];
    for (var, label) in [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("ANTHROPIC_AUTH_TOKEN", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("GEMINI_API_KEY", "gemini"),
        ("GOOGLE_API_KEY", "gemini"),
    ] {
        if let Ok(v) = std::env::var(var)
            && !v.trim().is_empty()
        {
            envs.push(format!("  {var} ({label}): {}", mask(v.trim())));
        }
    }
    if !envs.is_empty() {
        out.push_str("Dari variabel lingkungan (menimpa profil bawaan):\n");
        for e in envs {
            out.push_str(&e);
            out.push('\n');
        }
    }
    out
}

/// Sembunyikan kunci API sebelum perintah `/api ...` dicetak ke layar.
pub fn mask_command_display(line: &str) -> String {
    let Ok(words) = split_words(line) else {
        return line.into();
    };
    let Some(cmd) = words.first() else { return line.into() };
    if cmd != "/api" || !matches!(words.get(1).map(String::as_str), Some("set" | "session")) {
        return line.into();
    }
    let mut out = Vec::with_capacity(words.len());
    let mut i = 0;
    let mut pos_after_name = 0;
    while i < words.len() {
        let w = &words[i];
        if let Some(v) = w.strip_prefix("key=") {
            out.push(format!("key={}", mask(v)));
        } else if (w == "--key" || w == "-k") && i + 1 < words.len() {
            out.push(w.clone());
            i += 1;
            out.push(mask(&words[i]));
        } else if i >= 3 && !w.starts_with('-') && !w.contains('=') {
            if !w.starts_with("http://") && !w.starts_with("https://") && pos_after_name == 0 {
                pos_after_name += 1;
                out.push(mask(w));
            } else {
                out.push(w.clone());
            }
        } else {
            out.push(w.clone());
        }
        i += 1;
    }
    out.join(" ")
}

const API_HELP: &str = "Pengaturan API multi-provider:
  llmy api                                  daftar provider tersimpan
  llmy api set <nama> [opsi]                simpan provider ke ~/.config/llmy/api.json
    --protocol <ollama|anthropic|openai|gemini>
    --url <url>                             alamat endpoint
    --key <kunci>                           kunci API (ditanya tersembunyi bila kosong)
    --no-key                                tanpa kunci (mis. server lokal)
    --model <model>                         model bawaan untuk profil ini
    --default                               jadikan provider bawaan saat mulai
  llmy api use <nama> [model]               jadikan provider bawaan saat mulai
  llmy api remove <nama>                    hapus provider tersimpan

Contoh:
  llmy api set anthropic --key sk-ant-... --model claude-opus-5-5
  llmy api set gemini --key AIza... --model gemini-2.5-flash
  llmy api set deepseek --key sk-... --model deepseek-chat --default
  llmy api set vllm --protocol openai --url http://192.168.1.5:8000/v1 --no-key --model qwen";

fn prompt_secret(prompt: &str) -> Result<Option<String>, String> {
    eprint!("{prompt}");
    let _ = std::io::stderr().flush();
    terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    let mut buf = String::new();
    let res = loop {
        match event::read() {
            Ok(Event::Key(k)) if k.kind != KeyEventKind::Release => {
                let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                match k.code {
                    KeyCode::Enter => break Ok(Some(buf)),
                    KeyCode::Esc => break Ok(None),
                    KeyCode::Char('c') if ctrl => break Err("dibatalkan".into()),
                    KeyCode::Backspace => {
                        buf.pop();
                    }
                    KeyCode::Char(c) if !ctrl => buf.push(c),
                    _ => {}
                }
            }
            Ok(_) => {}
            Err(e) => break Err(e.to_string()),
        }
    };
    let _ = terminal::disable_raw_mode();
    eprintln!();
    res.map(|o| o.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
}

pub fn cli(args: &[String]) -> i32 {
    match run_cli(args) {
        Ok(msg) => {
            if !msg.is_empty() {
                print!("{msg}");
                if !msg.ends_with('\n') {
                    println!();
                }
            }
            0
        }
        Err(e) => {
            eprintln!("llmy api: {e}");
            2
        }
    }
}

fn run_cli(args: &[String]) -> Result<String, String> {
    let sub = args.first().map(String::as_str).unwrap_or("list");
    match sub {
        "-h" | "--help" | "help" => Ok(API_HELP.into()),
        "list" | "ls" => {
            let cfg = Config::load_strict()?;
            Ok(format_list(&cfg, &BTreeMap::new(), None))
        }
        "set" => {
            let name = args.get(1).ok_or("pakai: llmy api set <nama> [opsi]")?;
            let opts = parse_set_opts(&args[2..])?;
            if opts.session {
                return Err("--session hanya dipakai di mode interaktif (/api session)".into());
            }
            let mut cfg = Config::load_strict()?;
            let existing = cfg.providers.get(name).cloned();
            let set_default = opts.default;
            let mut p = build_profile(name, existing.as_ref(), opts)?;
            if p.protocol != Kind::Ollama && p.key.is_none() && std::io::stdin().is_terminal() {
                let url = p.url.as_deref().unwrap_or_else(|| default_url(p.protocol));
                if !crate::http::is_local(url)
                    && !crate::http::is_lan(url)
                    && let Some(k) = prompt_secret(&format!("Kunci API untuk {name} (Enter untuk kosong): "))?
                {
                    p.key = Some(k);
                }
            }
            let url = p.url.as_deref().unwrap_or_else(|| default_url(p.protocol)).to_string();
            let proto = kind_name(p.protocol);
            cfg.providers.insert(name.clone(), p);
            if set_default {
                cfg.default = Some(name.clone());
            }
            cfg.save()?;
            let def = if cfg.default.as_deref() == Some(name.as_str()) { " [bawaan]" } else { "" };
            Ok(format!("Provider '{name}' disimpan ({proto} · {url}){def}."))
        }
        "use" | "default" => {
            let name = args.get(1).ok_or("pakai: llmy api use <nama> [model]")?;
            let mut cfg = Config::load_strict()?;
            let model = args.get(2).cloned();
            if let Some(p) = cfg.providers.get_mut(name) {
                if let Some(m) = model {
                    p.model = Some(m);
                }
            } else if let Some((proto, url)) = preset(name) {
                cfg.providers.insert(name.clone(), Profile {
                    protocol: proto,
                    url: Some(url.into()),
                    key: None,
                    model,
                });
            } else {
                return Err(format!("provider '{name}' belum ada; simpan dulu dengan `llmy api set {name} ...`"));
            }
            cfg.default = Some(name.clone());
            cfg.save()?;
            let note = match cfg.providers.get(name).and_then(|p| p.model.as_deref()) {
                Some(m) => format!(" (model {m})"),
                None if name != "ollama" => " (belum ada model bawaan; pakai -m atau --model)".into(),
                None => String::new(),
            };
            Ok(format!("Provider bawaan sekarang '{name}'{note}."))
        }
        "remove" | "rm" | "delete" => {
            let name = args.get(1).ok_or("pakai: llmy api remove <nama>")?;
            let mut cfg = Config::load_strict()?;
            if cfg.providers.remove(name).is_none() {
                return Err(format!("provider '{name}' tidak ada"));
            }
            if cfg.default.as_deref() == Some(name.as_str()) {
                cfg.default = None;
            }
            cfg.save()?;
            Ok(format!("Provider '{name}' dihapus."))
        }
        _ => Err(format!("subperintah tidak dikenal: {sub} (lihat `llmy api --help`)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_kunci() {
        assert_eq!(mask("abc"), "••••");
        assert_eq!(mask("12345678"), "••••");
        assert_eq!(mask("sk-ant-1234567890"), "sk-a...7890");
    }

    #[test]
    fn mask_perintah_api() {
        assert_eq!(mask_command_display("/api set deepseek --key sk-1234567890"), "/api set deepseek --key sk-1...7890");
        assert_eq!(mask_command_display("/api set deepseek sk-1234567890"), "/api set deepseek sk-1...7890");
        assert_eq!(mask_command_display("/api set deepseek key=sk-1234567890"), "/api set deepseek key=sk-1...7890");
        assert_eq!(mask_command_display("/api use deepseek"), "/api use deepseek");
    }

    #[test]
    fn parse_opsi() {
        let w = split_words("set deepseek --key \"sk-abc def\" --model deepseek-chat --default").unwrap();
        let o = parse_set_opts(&w[2..]).unwrap();
        assert_eq!(o.key.as_deref(), Some("sk-abc def"));
        assert_eq!(o.model.as_deref(), Some("deepseek-chat"));
        assert!(o.default);
        assert!(parse_set_opts(&["ftp://x".into()]).is_err());
    }
}
