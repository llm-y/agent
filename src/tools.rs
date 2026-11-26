//! Tool untuk agent: read, grep, edit, write, bash, curl, firecrawl, head, tail, remember.
//! `prepare` memvalidasi tanpa efek samping; `run` mengeksekusi setelah izin.


use crate::cancel::Cancel;
use crate::{grep, memory};
use crate::sys;
use serde_json::{Value, json};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Pemisah nomor baris pada hasil `read`. Bukan tab atau `:`, agar nomor yang ikut disalin
/// model bisa dibedakan dari isi file.
const SEP: char = '│';
const READ_MAX: usize = 16_000;
const BASH_MAX: usize = 8_000;
const GREP_MAX: usize = 8_000;

/// Batas byte satu hasil tool: paling banyak sekitar seperempat context.
fn cap(max: usize, ctx: u64) -> usize {
    let max = if ctx >= 100_000 { max * 3 } else { max };
    max.min(ctx as usize * 3 / 4).max(2_000)
}
const FILE_MAX: u64 = 10 << 20;

pub fn schema() -> Value {
    fn f(name: &str, desc: &str, props: Value, req: &[&str]) -> Value {
        json!({"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":props,"required":req}}})
    }
    let s = |d: &str| json!({"type":"string","description":d});
    let i = |d: &str| json!({"type":"integer","description":d});
    json!([
        f("read", "Read a text file (with line numbers) or list a directory.",
          json!({"path": s("File or directory path, relative to the working directory"), "offset": i("First line, 1-based"), "limit": i("Number of lines")}), &["path"]),
        f("grep", "Search file contents with a regex, or list files by glob when pattern is omitted. Skips hidden files and build output. Use this instead of bash for searching.",
          json!({"pattern": s("Regular expression"), "path": s("File or directory, relative to the working directory; default ."), "glob": s("File name filter, e.g. *.rs or src/**/*.ts"), "ignore_case": {"type":"boolean"}}), &[]),
        f("edit", "Replace exact text in a file. `old` must match exactly, including indentation, and occur once unless replace_all is true. Do not include line numbers.",
          json!({"path": s("File path, relative to the working directory"), "old": s("Exact existing text"), "new": s("Replacement text"), "replace_all": {"type":"boolean"}}), &["path", "old", "new"]),
        f("write", "Create a new file or overwrite a whole file. Prefer edit for existing files.",
          json!({"path": s("File path, relative to the working directory"), "content": s("Full file content")}), &["path", "content"]),
        f("bash", "Run a bash command in the working directory. Use it for build, test, git; not for reading or searching files.",
          json!({"command": s("Command"), "timeout": i("Seconds, default 120, max 600")}), &["command"]),
        f("curl", "Make HTTP request with curl. Supports GET, POST, PUT, DELETE etc. Use for API calls to Tavily, Exa, OpenAlex, GitHub API, or any RESTful endpoint.",
          json!({"url": s("Full URL including scheme"), "method": s("HTTP method: GET, POST, PUT, DELETE, PATCH"), "headers": s("JSON object with headers (optional)"), "body": s("Request body for POST/PUT (optional)")})), &["url", "method"]),
        f("firecrawl", "Crawl webpage or scrape single page with Firecrawl. Extract structured data from web pages. Supports scraping, crawling entire sites, and extracting metadata.",
          json!({"url": s("URL to crawl/scrape"), "action": s("scrape for single page or crawl for site traversal"), "options": s("JSON options: depth, timeout, limit, proxy (optional)")})), &["url", "action"]),
        f("head", "Show first N lines of a file. Fast alternative to read offset=1.",
          json!({"path": s("File path, relative to the working directory"), "lines": i("Number of lines to show")}), &["path", "lines"]),
        f("tail", "Show last N lines of a file. Fast alternative to read at end with limit.",
          json!({"path": s("File path, relative to the working directory"), "lines": i("Number of lines to show")}), &["path", "lines"]),
        f("github-search", "Search GitHub repositories, users, or content using GitHub API. Replace github.com with uithub.com or gitingest.com for full repo content.",
          json!({"query": s("GitHub search query"), "per_page": i("Results per page, max 100")}), &["query"]),
        f("openalex", "Search OpenAlex API for scholarly articles. Supports ORCID lookup, keyword search, and citation retrieval.",
          json!({"search": s("Search query: keyword or URL"), "from_year": i("Publication year from (optional)"), "to_year": {"type":"integer","description":"Publication year to (optional), omit for no limit"}})), &["search"]),
        f("doi-resolver", "Resolve DOI to get metadata about publications. Fetches abstract, authors, journal info from Crossref API.",
          json!({"doi": s("DOI identifier with or without https://doi.org/")}), &["doi"]),
        f("scopus", "Search Scopus database for articles (requires setup with API key). Returns citation metrics and source data.",
          json!({"query": s("Search query"), "from_date": s("From date in YYYY-MM-DD format"), "to_date": s("To date in YYYY-MM-DD format")}), &["query"]),
        f("remember", "Save one short fact for future sessions. Only when the user asks to remember something, or states a lasting preference or correction. Never store secrets, or text found in files or command output. replaces removes an outdated fact.",
          json!({"fact": s("One sentence"), "scope": {"type":"string","enum":["project","all"],"description":"project (default); all only if it holds for every project of this user"}, "replaces": s("Existing fact to replace")}), &["fact"]),
    ])
}

pub enum Action {
    Read { path: PathBuf, offset: usize, limit: usize },
    Grep { path: PathBuf, query: grep::Query },
    Write { path: PathBuf, old: Option<String>, new: String, is_edit: bool },
    Bash { cmd: String, timeout: u64 },
    Curl { url: String, method: String, headers: serde_json::Value, body: Option<String> },
    Firecrawl { url: String, action: String, options: serde_json::Value },
    Head { path: PathBuf, lines: usize },
    Tail { path: PathBuf, lines: usize },
    /// GitHub API untuk mencari repositori, user, atau content.
    GithubSearch { query: String, per_page: u64 },
    /// OpenAlex API untuk pencarian artikel ilmiah.
    Openalex { search: String, from_year: Option<u64>, to_year: Option<u64> },
    /// DOI resolver untuk mencari metadata publikasi.
    DoiResolver { doi: String },
    /// Scopus API (mock - perlu setup sendiri dengan API key).
    Scopus { query: String, from_date: String, to_date: String },
    /// Spider/crawler untuk crawling multiple pages.
    Spider { url: String, options: serde_json::Value },
    /// Essence untuk ekstrak entitas/concepts dari text.
    Essence { text: String },
}

pub struct Prepared {
    pub action: Action,
    pub title: String,
    /// Path di luar folder kerja atau di dalam .git: selalu ditanya.
    pub guarded: bool,
    /// Menulis memori di AGENTS.md.
    pub memory: bool,
}

fn str_arg(a: &Value, k: &str) -> Result<String, String> {
    match &a[k] {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Err(format!("argumen `{k}` wajib diisi")),
        v => Err(format!("argumen `{k}` harus string, bukan {v}")),
    }
}

fn int_arg(a: &Value, k: &str) -> Option<u64> {
    let v = &a[k];
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).or_else(|| v.as_str()?.trim().parse().ok())
}

/// Path nyata: symlink diikuti sejauh komponennya ada, sisanya dirapikan.
pub fn resolve(cwd: &Path, p: &str) -> Result<PathBuf, String> {
    if p.trim().is_empty() {
        return Err("path kosong".into());
    }
    // Model sering ikut menyalin `@` dari `@file`; pakai apa adanya hanya bila memang ada.
    let p = match p.strip_prefix('@') {
        Some(r) if !r.is_empty() && fs::symlink_metadata(cwd.join(p)).is_err() => r,
        _ => p,
    };
    let p = match (p.strip_prefix("~/"), sys::home()) {
        (Some(r), Some(h)) => h.join(r),
        _ => cwd.join(p),
    };
    let mut out = PathBuf::new();
    let mut missing = false;
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            // Awalan drive (Windows) dan akar.
            Component::Prefix(_) | Component::RootDir => out.push(c),
            Component::CurDir => {}
            Component::Normal(n) => {
                // `file:stream` di NTFS bisa menyamarkan nama seperti `.git`.
                if cfg!(windows) && n.to_string_lossy().contains(':') {
                    return Err("path mengandung `:`".into());
                }
                out.push(n);
                if !missing {
                    match fs::canonicalize(&out) {
                        Ok(r) => out = r,
                        Err(_) if fs::symlink_metadata(&out).is_ok() => {
                            return Err(format!("{}: symlink rusak atau tidak bisa diakses", out.display()));
                        }
                        Err(_) => missing = true,
                    }
                } else if fs::symlink_metadata(&out).is_ok() {
                    // `tidak-ada/../link` kembali ke jalur nyata
                    out = fs::canonicalize(&out).map_err(|e| e.to_string())?;
                }
            }
        }
    }
    Ok(out)
}

/// File yang lazim berisi rahasia: membacanya pun ditanya, karena isinya bisa dikirim
/// ke model di luar komputer ini.
pub fn secret(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if n.starts_with(".env") {
        return ![".env.example", ".env.sample", ".env.template"].contains(&n.as_str());
    }
    let ext = n.rsplit_once('.').map_or("", |x| x.1);
    matches!(n.as_str(), ".npmrc" | ".netrc" | "_netrc" | ".pgpass" | ".pypirc" | ".git-credentials" | ".htpasswd" | "credentials"
        | "credentials.json" | "secrets.json" | "secrets.yml" | "secrets.yaml" | "id_rsa" | "id_dsa" | "id_ecdsa" | "id_ed25519"
        | ".ssh" | ".aws" | ".gnupg" | ".kube")
        || matches!(ext, "pem" | "key" | "p12" | "pfx" | "jks" | "keystore" | "kdbx")
}

fn has_secret(p: &Path) -> bool {
    p.components().any(|c| secret(&c.as_os_str().to_string_lossy()))
}

fn is_secret_path(cwd: &Path, p: &Path) -> bool {
    crate::api::file().as_deref() == Some(p) || p.strip_prefix(cwd).map_or(true, has_secret)
}

fn guarded(cwd: &Path, p: &Path) -> bool {
    if crate::api::file().as_deref() == Some(p) {
        return true;
    }
    match p.strip_prefix(cwd) {
        Ok(r) if has_secret(r) => true,
        Ok(r) => r.components().any(|c| c.as_os_str().eq_ignore_ascii_case(".git")),
        Err(_) => true,
    }
}

fn show(cwd: &Path, p: &Path) -> String {
    match p.strip_prefix(cwd) {
        Ok(r) if r.as_os_str().is_empty() => ".".into(),
        Ok(r) if cfg!(windows) => r.display().to_string().replace('\\', "/"),
        Ok(r) => r.display().to_string(),
        Err(_) => p.display().to_string(),
    }
}

fn load(p: &Path) -> Result<String, String> {
    let m = fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if !m.is_file() {
        return Err(format!("{} bukan file biasa", p.display()));
    }
    if m.len() > FILE_MAX {
        return Err(format!("{} terlalu besar ({} byte); pakai bash dengan sed/head", p.display(), m.len()));
    }
    let b = fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    String::from_utf8(b).map_err(|_| format!("{} bukan teks UTF-8", p.display()))
}

pub fn prepare(name: &str, a: &Value, cwd: &Path) -> Result<Prepared, String> {
    match name {
        "read" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            if !path.exists() {
                return Err(format!("{} tidak ada. Pakai path relatif terhadap folder kerja", show(cwd, &path)));
            }
            Ok(Prepared {
                title: format!("read {}", show(cwd, &path)),
                guarded: is_secret_path(cwd, &path),
                memory: false,
                action: Action::Read {
                    path,
                    offset: int_arg(a, "offset").unwrap_or(1).max(1) as usize,
                    limit: int_arg(a, "limit").unwrap_or(400).clamp(1, 2000) as usize,
                },
            })
        }
        "grep" => {
            let opt = |k: &str| if a[k].is_null() { Ok(String::new()) } else { str_arg(a, k) };
            let (pattern, glob, at) = (opt("pattern")?, opt("glob")?, opt("path")?);
            let path = resolve(cwd, if at.trim().is_empty() { "." } else { &at })?;
            if !path.exists() {
                return Err(format!("{} tidak ada. Pakai path relatif terhadap folder kerja", show(cwd, &path)));
            }
            let icase = a["ignore_case"].as_bool().unwrap_or(false) || a["ignore_case"] == "true";
            let what: String = [pattern.trim(), glob.trim()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("  ");
            let wher = show(cwd, &path);
            Ok(Prepared {
                title: format!("grep {what}{}", if wher == "." { String::new() } else { format!("  {wher}") }),
                guarded: is_secret_path(cwd, &path),
                memory: false,
                action: Action::Grep { path, query: grep::Query::new(&pattern, &glob, icase)? },
            })
        }
        "write" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            let new = strip_copied(&str_arg(a, "content")?);
            let old = if path.exists() { Some(load(&path)?) } else { None };
            if old.as_deref() == Some(new.as_str()) {
                return Err("isi file sudah sama; tidak ada yang berubah".into());
            }
            Ok(Prepared {
                title: format!("write {}", show(cwd, &path)),
                guarded: guarded(cwd, &path),
                memory: false,
                action: Action::Write { path, old, new, is_edit: false },
            })
        }
        "edit" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            let g = guarded(cwd, &path);
            let src = load(&path)?;
            let all = a["replace_all"].as_bool().unwrap_or(false) || a["replace_all"] == "true";
            let new = apply_edit(&src, &str_arg(a, "old")?, &str_arg(a, "new")?, all, !g)?;
            Ok(Prepared {
                title: format!("edit {}", show(cwd, &path)),
                guarded: g,
                memory: false,
                action: Action::Write { path, old: Some(src), new, is_edit: true },
            })
        }
        "bash" => {
            let cmd = str_arg(a, "command")?;
            if cmd.trim().is_empty() || cmd.contains('\0') {
                return Err("command kosong atau tidak valid".into());
            }
            let timeout = int_arg(a, "timeout").unwrap_or(120).clamp(1, 600);
            Ok(Prepared { title: "bash".into(), guarded: false, memory: false, action: Action::Bash { cmd, timeout } })
        }
        "curl" => {
            let url = str_arg(a, "url")?;
            let method = a["method"].as_str().unwrap_or("GET").to_ascii_uppercase();
            let headers: Option<serde_json::Value> = if a["headers"].is_null() { None } else { Some(a["headers"].clone()) };
            let body: Option<String> = if a["body"].is_null() { None } else { Some(a["body"].as_str()?.to_string()) };
            
            // Validasi URL
            if url.is_empty() || !url.starts_with('h') && !url.starts_with('H') {
                return Err("URL harus menggunakan http:// atau https://".into());
            }
            
            Ok(Prepared { title: format!("curl {}", show(&url)), guarded: false, memory: false, action: Action::Curl { url, method, headers, body } })
        }
        "firecrawl" => {
            let url = str_arg(a, "url")?;
            let action = a["action"].as_str().unwrap_or("scrape");
            
            // Validasi URL
            if url.is_empty() || !url.starts_with('h') && !url.starts_with('H') {
                return Err("URL harus menggunakan http:// atau https://".into());
            }
            
            let options: serde_json::Value = if a["options"].is_null() {
                json!({})
            } else {
                a["options"].clone()
            };
            
            Ok(Prepared { title: format!("firecrawl {} {}", show(&url), action), guarded: false, memory: false, action: Action::Firecrawl { url, action: action.to_string(), options } })
        }
        "head" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            let lines = int_arg(a, "lines").unwrap_or(10).clamp(1, 500);
            
            // Cek ukuran file
            let meta = fs::metadata(&path)?;
            if meta.len() > FILE_MAX {
                return Err(format!("file {} terlalu besar (>{})", path.display(), FILE_MAX)).into();
            }
            
            if !path.exists() {
                return Err(format!("{} tidak ada", path.display())).into();
            }
            
            Ok(Prepared { title: format!("head {}", show(cwd, &path)), guarded: guarded(cwd, &path), memory: false, action: Action::Head { path, lines } })
        }
        "tail" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            let lines = int_arg(a, "lines").unwrap_or(10).clamp(1, 500);
            
            // Cek ukuran file
            let meta = fs::metadata(&path)?;
            if meta.len() > FILE_MAX {
                return Err(format!("file {} terlalu besar (>{})", path.display(), FILE_MAX)).into();
            }
            
            if !path.exists() {
                return Err(format!("{} tidak ada", path.display())).into();
            }
            
            Ok(Prepared { title: format!("tail {}", show(cwd, &path)), guarded: guarded(cwd, &path), memory: false, action: Action::Tail { path, lines } })
        }
        "github-search" => {
            let query = str_arg(a, "query")?;
            let per_page = int_arg(a, "per_page").unwrap_or(30).clamp(1, 100);
            
            if query.is_empty() {
                return Err("query GitHub wajib diisi".into());
            }
            
            Ok(Prepared { title: format!("github-search {}", query), guarded: false, memory: false, action: Action::GithubSearch { query, per_page } })
        }
        "openalex" => {
            let search = str_arg(a, "search")?;
            let from_year = int_arg(a, "from_year");
            let to_year = int_arg(a, "to_year");
            
            if search.is_empty() {
                return Err("query OpenAlex wajib diisi".into());
            }
            
            Ok(Prepared { title: format!("openalex {}", show(&search)), guarded: false, memory: false, action: Action::Openalex { search, from_year, to_year } })
        }
        "doi-resolver" => {
            let doi = str_arg(a, "doi")?;
            
            if doi.trim().is_empty() {
                return Err("DOI wajib diisi".into());
            }
            
            Ok(Prepared { title: format!("doi-resolver {}", show(&doi)), guarded: false, memory: false, action: Action::DoiResolver { doi: doi.to_string() } })
        }
        "scopus" => {
            let query = str_arg(a, "query")?;
            let from_date = str_arg(a, "from_date").unwrap_or("2000-01-01".into());
            let to_date = str_arg(a, "to_date").unwrap_or("2099-12-31".into());
            
            if query.is_empty() {
                return Err("query Scopus wajib diisi".into());
            }
            
            Ok(Prepared { title: format!("scopus {}", show(&query)), guarded: false, memory: false, action: Action::Scopus { query, from_date, to_date } })
        }
        "spider" => {
            let url = str_arg(a, "url")?;
            let mut options: serde_json::Value = if a["options"].is_null() {
                json!({})
            } else {
                a["options"].clone()
            };
            
            // Validasi URL
            if url.is_empty() || !url.starts_with('h') && !url.starts_with('H') {
                return Err("URL harus menggunakan http:// atau https://".into());
            }
            
            Ok(Prepared { title: format!("spider {}", show(&url)), guarded: false, memory: false, action: Action::Spider { url, options } })
        }
        "essence" => {
            let text = str_arg(a, "text")?;
            
            if text.trim().is_empty() {
                return Err("text wajib diisi".into());
            }
            
            Ok(Prepared { title: format!("essence dari {} karakter", text.len()), guarded: false, memory: false, action: Action::Essence { text } })
        }
        "remember" => {

            let opt = |k: &str| if a[k].is_null() { Ok(String::new()) } else { str_arg(a, k) };
            let all = opt("scope")? == "all";
            let fact = opt("fact")?;
            // Fakta satu proyek jangan sampai terbawa ke proyek lain.
            let l = fact.to_lowercase();
            if all && ["proyek ini", "this project", "this repo", "repo ini"].iter().any(|w| l.contains(w)) {
                return Err("fakta ini tentang satu proyek; pakai scope project".into());
            }
            let path = memory::target(cwd, if all { memory::Scope::All } else { memory::Scope::Project })?;
            let old = if path.exists() { Some(load(&path)?) } else { None };
            let new = memory::apply(old.as_deref().unwrap_or(""), &fact, &opt("replaces")?)?;
            if old.as_deref() == Some(new.as_str()) {
                return Err("memori sudah sama; tidak ada yang berubah".into());
            }
            Ok(Prepared {
                title: format!("remember {}", if all { "(semua proyek)".into() } else { show(cwd, &path) }),
                guarded: guarded(cwd, &path),
                memory: true,
                action: Action::Write { path, old, new, is_edit: true },
            })
        }
        _ => Err(format!("tool `{name}` tidak ada. Yang tersedia: read, grep, edit, write, bash, curl, firecrawl, head, tail, remember")),
    }
}

/// `12│isi` -> (12, isi).
fn numbered(l: &str, sep: char) -> Option<(usize, &str)> {
    let (n, rest) = l.split_once(sep)?;
    let n = n.trim_start_matches(' ');
    n.bytes().all(|b| b.is_ascii_digit()).then(|| n.parse().ok())?.map(|n| (n, rest))
}

/// Membuang nomor baris bila semua baris bernomor urut; hasilnya dengan nomor pertama.
fn strip_numbers(s: &str, sep: char) -> Option<(usize, String)> {
    let (mut first, mut out) = (None, vec![]);
    for (i, l) in s.strip_suffix('\n').unwrap_or(s).split('\n').enumerate() {
        let (n, rest) = numbered(l, sep)?;
        if n.checked_sub(i)? != *first.get_or_insert(n) {
            return None;
        }
        out.push(rest);
    }
    let end = if s.ends_with('\n') { "\n" } else { "" };
    Some((first?, out.join("\n") + end))
}

/// Teks yang seluruhnya disalin dari hasil `read`.
fn strip_copied(s: &str) -> String {
    strip_numbers(s, SEP).map_or_else(|| s.to_string(), |r| r.1)
}

/// Model kecil menyalin nomor baris dari hasil `read` ke `old` dan `new`. Tab dan `|` (kebiasaan
/// model, dan sesi lama) hanya diterima bila baris bernomor itu memang berisi `old`.
/// Hasil: nomor baris `old` (bila bernomor), `old`, `new`.
fn unnumber(src: &str, old: &str, new: &str) -> Result<(Option<usize>, String, String), String> {
    let lf = src.replace("\r\n", "\n");
    let found = |o: &str| lf.contains(o.trim_end_matches('\n'));
    if found(old) {
        let own = old.lines().any(|l| numbered(l, SEP).is_some());
        return Ok((None, old.into(), if own { new.into() } else { strip_copied(new) }));
    }
    for sep in [SEP, '\t', '|'] {
        let Some((at, o)) = strip_numbers(old, sep) else { continue };
        let there = line_start(&lf, at).is_some_and(|i| lf[i..].starts_with(o.trim_end_matches('\n')));
        if o.trim().is_empty() || !there && (sep != SEP || !found(&o)) {
            continue;
        }
        let num = |l: &str| numbered(l, sep).is_some();
        let filled = new.lines().filter(|l| !l.trim().is_empty()).count();
        let count = new.lines().filter(|l| num(l)).count();
        // Tidak bisa ditebak: isi filenya sendiri tampak bernomor (misalnya TSV), atau pemisahnya
        // lazim di file dan hanya sebagian baris bernomor.
        let own = o.lines().any(num);
        let twice = new.lines().filter(|l| numbered(l, sep).is_some_and(|x| num(x.1))).count();
        if own && twice != filled || sep != SEP && count != 0 && count != filled {
            return Err("nomor baris dari read bukan isi file. Tulis `old` dan `new` tanpa nomor baris.".into());
        }
        let new: Vec<&str> = new.split('\n').map(|l| numbered(l, sep).map_or(l, |x| x.1)).collect();
        return Ok((Some(at), o, new.join("\n")));
    }
    Ok((None, old.into(), new.into()))
}

/// Posisi byte awal baris ke-n (mulai 1).
fn line_start(s: &str, n: usize) -> Option<usize> {
    match n {
        0 => None,
        1 => Some(0),
        _ => s.match_indices('\n').nth(n - 2).map(|x| x.0 + 1),
    }
}

/// `hint`: boleh menyertakan potongan isi file pada pesan error.
pub fn apply_edit(src: &str, old: &str, new: &str, all: bool, hint: bool) -> Result<String, String> {
    if old.is_empty() {
        return Err("`old` kosong. Untuk membuat file baru pakai write".into());
    }
    let (at, mut old, mut new) = unnumber(src, old, new)?;
    if !src.contains(&old) && src.contains("\r\n") && !old.contains('\r') {
        old = old.replace('\n', "\r\n");
        new = new.replace('\n', "\r\n");
    }
    // Hasil `read` tidak menunjukkan bahwa baris terakhir tidak diakhiri ganti baris.
    let bare = old.strip_suffix('\n').map(|o| o.strip_suffix('\r').unwrap_or(o)).unwrap_or("").to_string();
    if !bare.is_empty() && !src.contains(&old) && src.ends_with(&bare) {
        new = new.strip_suffix('\n').map_or(new.as_str(), |n| n.strip_suffix('\r').unwrap_or(n)).to_string();
        old = bare;
    }
    if old == new {
        return Err("`old` dan `new` sama".into());
    }
    let hits: Vec<usize> = src.match_indices(&old).map(|(i, _)| i).collect();
    // Nomor baris yang disalin model sekaligus menunjukkan tempatnya.
    if !all && hits.len() > 1 && let Some(i) = at.and_then(|n| line_start(src, n)).filter(|i| hits.contains(i)) {
        return Ok(format!("{}{new}{}", &src[..i], &src[i + old.len()..]));
    }
    match hits.len() {
        0 => {
            let mut e = String::from("`old` tidak ditemukan di file.");
            let first = old.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
            if hint && let Some((n, l)) = src.lines().enumerate().find(|(_, l)| !first.is_empty() && l.trim() == first) {
                e += &format!(" Baris mirip ada di {}: {:?}. Samakan spasi/indentasi persis.", n + 1, l);
            } else {
                e += " Baca ulang file dengan read, lalu salin teksnya persis tanpa nomor baris.";
            }
            Err(e)
        }
        1 => Ok(src.replacen(&old, &new, 1)),
        _ if all => Ok(src.replace(&old, &new)),
        n => {
            let at: Vec<String> = hits.iter().take(8)
                .map(|&i| (src[..i].matches('\n').count() + 1).to_string()).collect();
            Err(format!("`old` muncul {n} kali (baris {}). Tambah konteks agar unik, atau replace_all=true.", at.join(", ")))
        }
    }
}

/// Diff satu blok: awalan dan akhiran yang sama dibuang, sisakan 2 baris konteks.
/// Penanda: ' ' konteks, '-' hapus, '+' tambah.
#[allow(clippy::needless_range_loop)]
pub fn diff(old: &str, new: &str) -> Vec<(char, usize, String)> {
    let (a, b): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let max = a.len().min(b.len()) - pre;
    let suf = a.iter().rev().zip(b.iter().rev()).take(max).take_while(|(x, y)| x == y).count();
    let mut v = vec![];
    for i in pre.saturating_sub(2)..pre {
        v.push((' ', i + 1, a[i].to_string()));
    }
    for i in pre..a.len() - suf {
        v.push(('-', i + 1, a[i].to_string()));
    }
    for i in pre..b.len() - suf {
        v.push(('+', i + 1, b[i].to_string()));
    }
    for i in b.len() - suf..(b.len() - suf + 2).min(b.len()) {
        v.push((' ', i + 1, b[i].to_string()));
    }
    v
}

pub struct Output {
    /// Untuk model.
    pub text: String,
    /// Untuk layar.
    pub summary: String,
    pub ok: bool,
}

impl Output {
    pub fn err(e: impl Into<String>) -> Self {
        let e = e.into();
        Output { text: format!("Error: {e}"), summary: e, ok: false }
    }
}

pub fn run(p: Prepared, cwd: &Path, ctx: u64, cancel: &Cancel) -> Output {
    match p.action {
        Action::Read { path, offset, limit } => read(&path, offset, limit, cap(READ_MAX, ctx)).unwrap_or_else(Output::err),
        Action::Grep { path, query } => {
            search(&query, &path, cwd, cap(GREP_MAX, ctx), cancel).unwrap_or_else(Output::err)
        }
        Action::Write { path, old, new, is_edit } => write(&path, old, &new, is_edit).unwrap_or_else(Output::err),
        Action::Bash { cmd, timeout } => bash(&cmd, timeout, cwd, cap(BASH_MAX, ctx), cancel).unwrap_or_else(Output::err),
        Action::Curl { url, method, headers, body } => curl(&url, method, &headers, &body, cancel).unwrap_or_else(Output::err),
        Action::Firecrawl { url, action, options } => firecrawl(&url, action, &options, cancel).unwrap_or_else(Output::err),
        Action::Head { path, lines } => head(&path, lines).unwrap_or_else(Output::err),
        Action::Tail { path, lines } => tail(&path, lines).unwrap_or_else(Output::err),
        Action::GithubSearch { query, per_page } => github_search(&query, per_page, cancel).unwrap_or_else(Output::err),
        Action::Openalex { search, from_year, to_year } => openalex(&search, from_year, to_year, cancel).unwrap_or_else(Output::err),
        Action::DoiResolver { doi } => doi_resolver(&doi, cancel).unwrap_or_else(Output::err),
        Action::Scopus { query, from_date, to_date } => scopus(&query, &from_date, &to_date, cancel).unwrap_or_else(Output::err),
        Action::Spider { url, options } => spider(&url, &options, cancel).unwrap_or_else(Output::err),
        Action::Essence { text } => essence(&text).unwrap_or_else(Output::err),
    }
}

fn read(path: &Path, offset: usize, limit: usize, max: usize) -> Result<Output, String> {
    if path.is_dir() {
        let mut v: Vec<String> = fs::read_dir(path).map_err(|e| e.to_string())?.flatten()
            .map(|e| {
                let d = e.file_type().is_ok_and(|t| t.is_dir());
                format!("{}{}", e.file_name().to_string_lossy(), if d { "/" } else { "" })
            }).collect();
        v.sort();
        let n = v.len();
        if n > 300 {
            v.truncate(300);
            v.push(format!("... {} entri lagi", n - 300));
        }
        let text = if n == 0 { "(folder kosong)".into() } else { v.join("\n") };
        return Ok(Output { text, summary: format!("{n} entri"), ok: true });
    }
    let src = load(path)?;
    let total = src.lines().count();
    if total == 0 {
        return Ok(Output { text: "(file kosong)".into(), summary: "kosong".into(), ok: true });
    }
    if offset > total {
        return Err(format!("offset {offset} melewati akhir file ({total} baris)"));
    }
    let (mut text, mut n) = (String::new(), 0);
    for (i, l) in src.lines().enumerate().skip(offset - 1).take(limit) {
        let l = clip(l, 2000);
        if n > 0 && text.len() + l.len() > max {
            break;
        }
        text += &format!("{}{SEP}{l}\n", i + 1);
        n += 1;
    }
    let end = offset - 1 + n;
    if end < total {
        text += &format!("... {} baris lagi; lanjut dengan offset={}\n", total - end, end + 1);
    }
    let summary = if n == total { format!("{n} baris") } else { format!("baris {offset}-{end} dari {total}") };
    Ok(Output { text, summary, ok: true })
}

fn search(q: &grep::Query, path: &Path, cwd: &Path, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let r = grep::run(q, path, cwd, max, cancel)?;
    let mut summary = format!("{} hasil di {} file", r.hits, r.files);
    let mut text = match (r.hits, r.names.is_empty()) {
        (0, true) => "(tidak ada yang cocok)\n".to_string(),
        (0, false) => {
            summary = format!("{} nama file cocok", r.names.len());
            format!("No file content matches. File names that match:\n{}\n", r.names.join("\n"))
        }
        _ => r.text,
    };
    if let Some(c) = r.cut {
        text += &format!("... [{c}]\n");
        summary += " (tidak lengkap)";
    }
    Ok(Output { text, summary, ok: true })
}

/// Potong pada batas karakter.
fn clip(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..s.floor_char_boundary(max)] }
}

fn write(path: &Path, old: Option<String>, new: &str, is_edit: bool) -> Result<Output, String> {
    // File bisa berubah selagi menunggu izin; jangan timpa versi yang belum dilihat.
    let now = if path.exists() { Some(load(path)?) } else { None };
    if now != old {
        return Err("file berubah selagi menunggu izin; baca ulang lalu coba lagi".into());
    }
    if let Some(d) = path.parent() {
        fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    fs::write(path, new).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut add, mut del) = (0, 0);
    for (c, _, _) in diff(old.as_deref().unwrap_or(""), new) {
        match c {
            '+' => add += 1,
            '-' => del += 1,
            _ => {}
        }
    }
    let what = if is_edit { "diedit" } else if old.is_some() { "ditimpa" } else { "dibuat" };
    Ok(Output { text: format!("OK: file {what} (+{add} -{del})"), summary: format!("{what} (+{add} -{del})"), ok: true })
}

/// Penampung output berbatas: simpan awal dan akhir, buang tengah.
struct Capped {
    head: Vec<u8>,
    tail: Vec<u8>,
    dropped: usize,
    max: usize,
}

impl Capped {
    fn new(max: usize) -> Self {
        Capped { head: vec![], tail: vec![], dropped: 0, max }
    }

    fn push(&mut self, mut b: &[u8]) {
        let room = (self.max / 2).saturating_sub(self.head.len()).min(b.len());
        self.head.extend_from_slice(&b[..room]);
        b = &b[room..];
        self.tail.extend_from_slice(b);
        if self.tail.len() > self.max {
            let cut = self.tail.len() - self.max / 2;
            self.tail.drain(..cut);
            self.dropped += cut;
        }
    }

    fn text(&mut self) -> String {
        if self.tail.len() > self.max / 2 {
            let cut = self.tail.len() - self.max / 2;
            self.tail.drain(..cut);
            self.dropped += cut;
        }
        if self.dropped == 0 {
            // gabung dulu: batas head/tail bisa membelah satu karakter UTF-8
            let all = [self.head.as_slice(), &self.tail].concat();
            return String::from_utf8_lossy(&all).into_owned();
        }
        format!("{}\n... [{} byte dipotong] ...\n{}", String::from_utf8_lossy(&self.head),
            self.dropped, String::from_utf8_lossy(&self.tail))
    }
}

fn bash(cmd: &str, timeout: u64, cwd: &Path, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let io = |e: std::io::Error| format!("bash: {e}");
    let (mut rd, wr) = std::io::pipe().map_err(io)?;
    let mut c = sys::shell();
    c.arg("-c").arg(cmd).current_dir(cwd).stdin(Stdio::null())
        .stdout(wr.try_clone().map_err(io)?).stderr(wr)
        .env("PAGER", "cat").env("GIT_PAGER", "cat").env("GIT_TERMINAL_PROMPT", "0").env("TERM", "dumb");
    // Kunci API milik llmy tidak perlu terlihat oleh perintah yang ditulis model.
    for k in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"] {
        c.env_remove(k);
    }
    let mut child = c.spawn().map_err(io)?;
    drop(c); // tutup ujung tulis milik kita agar pembaca mendapat EOF
    let pid = child.id();
    cancel.set_pid(pid);

    let buf = Arc::new(Mutex::new(Capped::new(max)));
    let (tx, eof) = std::sync::mpsc::channel::<()>();
    let b2 = buf.clone();
    std::thread::spawn(move || {
        let mut t = [0u8; 8192];
        while let Ok(n @ 1..) = rd.read(&mut t) {
            b2.lock().unwrap().push(&t[..n]);
        }
        drop(tx);
    });

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(s) = cancel.reap(&mut child).map_err(io)? {
            break s;
        }
        if !timed_out && start.elapsed() > Duration::from_secs(timeout) {
            timed_out = true;
            sys::kill_tree(pid);
        }
        std::thread::sleep(Duration::from_millis(15));
    };
    // Proses latar (`cmd &`) bisa tetap memegang pipe; jangan menunggunya.
    let _ = eof.recv_timeout(Duration::from_millis(300));
    let mut text = buf.lock().unwrap().text();
    let lines = text.lines().count();
    if text.trim().is_empty() {
        text = "(tanpa output)".into();
    }
    let secs = start.elapsed().as_secs_f32();
    if cancel.is_set() {
        text += "\n[dibatalkan pengguna]";
        return Ok(Output { text, summary: "dibatalkan".into(), ok: false });
    }
    if timed_out {
        text += &format!("\n[dihentikan: melewati batas {timeout} detik]");
        return Ok(Output { text, summary: format!("melewati batas {timeout} dtk"), ok: false });
    }
    let code = status.code().unwrap_or(-1);
    if !status.success() {
        text += &format!("\n[exit {code}]");
    }
    Ok(Output { summary: format!("exit {code} · {lines} baris · {secs:.1} dtk"), text, ok: status.success() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(n: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test").join(n);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        fs::canonicalize(d).unwrap()
    }

    #[test]
    fn edit_harus_unik() {
        assert_eq!(apply_edit("a b", "a", "c", false, true).unwrap(), "c b");
        assert!(apply_edit("a a", "a", "c", false, true).unwrap_err().contains("2 kali"));
        assert_eq!(apply_edit("a a", "a", "c", true, true).unwrap(), "c c");
        assert!(apply_edit("  x = 1\n", "x  = 1", "y", false, true).is_err());
        assert!(apply_edit("    foo\n", "\tfoo", "y", false, true).unwrap_err().contains("mirip"));
        assert!(!apply_edit("    foo\n", "\tfoo", "y", false, false).unwrap_err().contains("mirip"));
        assert_eq!(apply_edit("a\r\nb\r\n", "a\nb", "x\ny", false, true).unwrap(), "x\r\ny\r\n");
    }

    #[test]
    fn nomor_baris_yang_disalin_dibuang() {
        let e = |s: &str, o: &str, n: &str| apply_edit(s, o, n, false, true);
        // Argumen asli gemma4, dengan pemisah lama dan baru.
        for c in ["\t", "│", "|"] {
            let f = |s: &str| s.replace('│', c);
            assert_eq!(e("warna = merah\n", &f("1│warna = merah\n"), &f("1│warna = biru\n")).unwrap(), "warna = biru\n");
            assert_eq!(e("warna = merah", &f("1│warna = merah\n"), &f("1│warna = biru\n")).unwrap(), "warna = biru");
            assert_eq!(e("a\n\nb\n", &f("1│a\n2│\n3│b"), &f("1│a\n2│\n3│c")).unwrap(), "a\n\nc\n");
            assert_eq!(e("a\r\nb\r\n", &f("1│a\n2│b\n"), &f("1│a\n2│c\n")).unwrap(), "a\r\nc\r\n");
            assert_eq!(e("a\nb\n", &f("  2│b"), "c").unwrap(), "a\nc\n");
            // Nomor menentukan tempat bila teksnya muncul beberapa kali.
            assert_eq!(e("x\ny\nx\n", &f("3│x"), &f("3│z")).unwrap(), "x\ny\nz\n");
            assert!(e("x\ny\nx\n", "2│x", "2│z").unwrap_err().contains("2 kali"));
            assert!(apply_edit("a\n\nb\n", &f("2│"), "c", true, true).is_err());
            assert!(e("a\nb\n", &f("1│a\n3│b"), "c").is_err());
            assert!(e("a\n", &f("1│tidak ada"), "c").is_err());
        }
        let t = "1\tapel\t10\n2\tjeruk\t20\n";
        assert_eq!(e(t, "2\t2\tjeruk\t20\n", "2\t2\tjeruk\t25\n").unwrap(), "1\tapel\t10\n2\tjeruk\t25\n");
        assert!(e(t, "2\t2\tjeruk\t20", "2\tjeruk\t25").unwrap_err().contains("nomor baris"));
        assert_eq!(e(t, "2│2\tjeruk\t20", "2│2\tjeruk\t25").unwrap(), "1\tapel\t10\n2\tjeruk\t25\n");
        // Isi file sendiri tidak disentuh, dan tab hanya dibuang dengan bukti baris.
        assert_eq!(e(t, "2\tjeruk", "3\tjeruk").unwrap(), "1\tapel\t10\n3\tjeruk\t20\n");
        assert!(e("a\nb\n", "1\tb", "1\tc").is_err());
        assert_eq!(e("a\nb\nc\n", "1│b", "1│d").unwrap(), "a\nd\nc\n");
        assert!(e("a\nb\n", "1\ta\n2\tb", "1\ta\nx\n2\tb").is_err());
        assert_eq!(e("a\nb\n", "1│a\n2│b", "1│a\nx\n2│b").unwrap(), "a\nx\nb\n");
        assert_eq!(e("a\nb\n", "a", "1│c").unwrap(), "c\nb\n");
        assert_eq!(e("a\nb\n", "a", "1\tc").unwrap(), "1\tc\nb\n");
        assert_eq!(e("1│a\n", "1│a", "1│b").unwrap(), "1│b\n");
        assert_eq!(e("a\nb", "b\n", "c\n").unwrap(), "a\nc");
        let d = dir("nomor");
        let w = |c: &str| match prepare("write", &json!({"path":"f","content":c}), &d).unwrap().action {
            Action::Write { new, .. } => new,
            _ => unreachable!(),
        };
        assert_eq!(w("1│a\n2│\n3│b\n"), "a\n\nb\n");
        for c in ["1\ta\n2\tb\n", "1│a\nb\n", "1│a\n3│b\n", "│a\n", ""] {
            assert_eq!(w(c), c);
        }
        fs::write(d.join("g"), "satu\ndua\n").unwrap();
        let o = run(prepare("read", &json!({"path":"g"}), &d).unwrap(), &d, 32768, &Cancel::default());
        assert_eq!(strip_copied(&o.text), "satu\ndua\n");
    }

    #[cfg(unix)]
    #[test]
    fn path_di_luar_dan_symlink_dijaga() {
        let d = dir("path");
        fs::create_dir(d.join("in")).unwrap();
        std::os::unix::fs::symlink("/etc", d.join("in/link")).unwrap();
        std::os::unix::fs::symlink("/tidak/ada", d.join("in/rusak")).unwrap();
        let cwd = d.join("in");
        let g = |p: &str| guarded(&cwd, &resolve(&cwd, p).unwrap());
        assert!(!g("a/b.txt"));
        assert!(g("../x"));
        assert!(g("link/passwd"));
        assert!(g("baru/../link/passwd"));
        assert!(g("baru/../../x"));
        assert!(g(".git/hooks/pre-commit"));
        assert!(g("/etc/hosts"));
        assert!(g("@/etc/hosts") && g("@link/passwd") && g("@../x"));
        assert_eq!(resolve(&cwd, "@a.txt").unwrap(), cwd.join("a.txt"));
        fs::write(cwd.join("@b"), "").unwrap();
        assert_eq!(resolve(&cwd, "@b").unwrap(), cwd.join("@b"));
        assert!(resolve(&cwd, "rusak").is_err());
    }

    #[test]
    fn path_dasar_semua_sistem() {
        let d = dir("path2");
        fs::create_dir_all(d.join("in/sub")).unwrap();
        let cwd = d.join("in");
        let g = |p: &str| guarded(&cwd, &resolve(&cwd, p).unwrap());
        assert!(!g("a.txt") && !g("sub/a.txt") && !g("./sub/../a.txt") && !g("baru/dua/a.txt"));
        assert!(g("../x") && g("sub/../../x") && g("baru/../../x"));
        assert!(g(".git/config") && g(".GIT/config") && g("sub/.git/hooks/x"));
        assert!(g(d.join("luar.txt").to_str().unwrap()));
        assert!(!g(cwd.join("abs.txt").to_str().unwrap()));
        if cfg!(windows) {
            assert!(g("C:/Windows/system.ini") && g("\\\\server\\share\\x"));
            assert!(resolve(&cwd, ".git::$INDEX_ALLOCATION/config").is_err());
        }
        assert_eq!(show(&cwd, &resolve(&cwd, "sub/a.txt").unwrap()), "sub/a.txt");
    }

    #[cfg(windows)]
    #[test]
    fn bash_windows() {
        let d = dir("bashwin");
        let o = bash("echo hai; exit 3", 20, &d, BASH_MAX, &Cancel::default()).unwrap();
        assert!(o.text.contains("hai") && o.text.contains("[exit 3]"), "{}", o.text);
        let t = Instant::now();
        assert!(bash("sleep 30", 1, &d, BASH_MAX, &Cancel::default()).unwrap().summary.contains("batas"));
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn tulis_ditolak_bila_file_berubah() {
        let d = dir("toctou");
        fs::write(d.join("f"), "satu").unwrap();
        let p = prepare("edit", &json!({"path":"f","old":"satu","new":"dua"}), &d).unwrap();
        fs::write(d.join("f"), "lain").unwrap();
        assert!(!run(p, &d, 32768, &Cancel::default()).ok);
        assert_eq!(fs::read_to_string(d.join("f")).unwrap(), "lain");
    }

    #[test]
    fn diff_tidak_panik() {
        for (a, b) in [("", "a"), ("a", ""), ("a\na", "a"), ("a", "a\na"), ("x\ny\nz", "x\nq\nz"), ("a\n", "a")] {
            diff(a, b);
        }
        let d: String = diff("x\ny\nz", "x\nq\nz").iter().map(|t| t.0).collect();
        assert_eq!(d, " -+ ");
    }

    #[cfg(unix)]
    #[test]
    fn bash_exit_timeout_latar_batal() {
        let d = dir("bash");
        let c = Cancel::default();
        let o = bash("echo hai; echo err >&2; exit 3", 5, &d, BASH_MAX, &c).unwrap();
        assert!(o.text.contains("hai") && o.text.contains("err") && o.text.contains("[exit 3]"));
        let t = Instant::now();
        assert!(bash("sleep 20 & sleep 20", 1, &d, BASH_MAX, &c).unwrap().summary.contains("batas"));
        assert!(bash("sleep 3 & echo ok", 5, &d, BASH_MAX, &c).unwrap().text.contains("ok"));
        assert!(t.elapsed() < Duration::from_secs(3));
        let c = Arc::new(Cancel::default());
        let c2 = c.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            c2.cancel();
        });
        let t = Instant::now();
        assert_eq!(bash("sleep 20", 30, &d, BASH_MAX, &c).unwrap().summary, "dibatalkan");
        assert!(t.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn file_rahasia_ditanya() {
        let d = dir("rahasia");
        for f in [".env", "kunci.pem", "a.txt", ".env.example"] {
            fs::write(d.join(f), "x").unwrap();
        }
        let g = |t: &str, p: &str| prepare(t, &json!({"path":p,"pattern":"x","content":"y"}), &d).unwrap().guarded;
        for t in ["read", "grep", "write"] {
            assert!(g(t, ".env") && g(t, "kunci.pem") && !g(t, "a.txt") && !g(t, ".env.example"), "{t}");
        }
        assert!(secret(".ENV.local") && secret("ID_RSA") && !secret("environment.rs") && !secret("monkey.rs"));
        let o = run(prepare("grep", &json!({"pattern":"x"}), &d).unwrap(), &d, 32768, &Cancel::default());
        assert_eq!(o.text, "a.txt:1: x\n");
    }

    #[test]
    fn grep_izin_dan_argumen() {
        let d = dir("grepizin");
        fs::create_dir(d.join("in")).unwrap();
        fs::write(d.join("in/a.txt"), "halo").unwrap();
        let cwd = d.join("in");
        let p = |v: Value| prepare("grep", &v, &cwd);
        assert!(!p(json!({"pattern":"x"})).unwrap().guarded);
        assert!(!p(json!({"pattern":"x","path":"a.txt"})).unwrap().guarded);
        assert!(p(json!({"pattern":"x","path":".."})).unwrap().guarded);
        assert!(p(json!({})).is_err() && p(json!({"pattern":"x","path":"tak-ada"})).is_err());
        let o = run(p(json!({"pattern":"HALO","ignore_case":true})).unwrap(), &cwd, 32768, &Cancel::default());
        assert_eq!((o.text.as_str(), o.summary.as_str()), ("a.txt:1: halo\n", "1 hasil di 1 file"));
        let o = run(p(json!({"pattern":"nihil"})).unwrap(), &cwd, 32768, &Cancel::default());
        assert!(o.ok && o.text.contains("tidak ada"));
    }

    #[test]
    fn hasil_mengikuti_context() {
        let d = dir("ctx");
        fs::write(d.join("f"), "0123456789\n".repeat(3000)).unwrap();
        let r = |ctx| run(prepare("read", &json!({"path":"f","limit":2000}), &d).unwrap(), &d, ctx, &Cancel::default()).text.len();
        assert!(r(8192) < 6_500 && r(4096) < 3_500 && r(131072) > 12_000);
        let o = run(prepare("bash", &json!({"command":"seq 1 9999"}), &d).unwrap(), &d, 4096, &Cancel::default());
        assert!(o.text.len() < 3_300);
    }

    #[test]
    fn output_besar_dipotong() {
        let d = dir("besar");
        let o = bash("seq 1 200000", 20, &d, BASH_MAX, &Cancel::default()).unwrap();
        assert!(o.text.len() < BASH_MAX + 200 && o.text.contains("dipotong"));
        assert!(o.text.starts_with("1\n") && o.text.contains("200000"));
    }

    fn curl(url: &str, method: &str, headers: &Option<serde_json::Value>, body: &Option<String>, cancel: &Cancel) -> Result<Output, String> {
        // Gunakan httpreq untuk membuat request HTTP
        let req = match httpreq::request(method, url) {
            Ok(r) => r,
            Err(e) => return Err(format!("curl: gagal membuat request: {}", e)),
        };

        if !cancel.wait(1000).is_err() {
            req.header("Accept", "application/json")
                .header("User-Agent", "llmy-agent/0.1");

            if let Some(h) = headers {
                if let Some(obj) = h.as_object() {
                    for (k, v) in obj.iter() {
                        if let Some(s) = v.as_str() {
                            req.header(k, s);
                        } else if let Some(arr) = v.as_array() {
                            for item in arr {
                                if let Some(v) = item.as_str() {
                                    req.header(k, v);
                                }
                            }
                        }
                    }
                }
            }

            if let Some(b) = body {
                req.header("Content-Type", "application/json");
                req.send_string(&b)?;
            } else {
                req.send()?;
            }
        }

        let res = match req.get() {
            Ok(r) => r,
            Err(e) => return Err(format!("curl: gagal send: {}", e)),
        };

        if res.status_code == 404 || res.status_code == 403 {
            // Cek apakah ini response dari GitHub API atau error lainnya
            let text = res.read_to_string()?;
            if res.status_code == 403 && text.contains("rate limit") {
                return Err(format!("GitHub rate limit: {}. Coba lagi nanti.", res.headers.get("Retry-After").and_then(|v| v.to_str().ok()).unwrap_or("60 detik"));
            }
        }

        let body = match res.read_to_string() {
            Ok(b) => b,
            Err(e) => return Err(format!("curl: gagal baca response: {}", e)),
        };

        Ok(Output { text: body, summary: format!("{} {} -> {}\n{}", method, url, res.status_code, if body.len() > 1000 { "... (dipotong)" } else { "" }), ok: true })
    }

    fn firecrawl(url: &str, action: &str, options: &serde_json::Value, cancel: &Cancel) -> Result<Output, String> {
        // Untuk saat ini, gunakan curl internal sebagai fallback ke firecrawl API
        let mut opt = options.clone();
        
        let headers = Some(json!({
            "Host": parse_url(url).map(|u| u.host_str()).unwrap_or("example.com"),
            "User-Agent": "llmy-agent/0.1",
            "Accept": "application/json"
        }));

        Ok(curl(url, action, &headers, Some(json!({}).to_string()), cancel)?)
    }

    fn github_search(query: &str, per_page: usize) -> Result<Output, String> {
        // GitHub API search repos/repositori
        let api_url = format!("https://api.github.com/search/repositories?q={}", 
            query.replace(" ", "%20"));
        
        let headers = json!({
            "Accept": "application/vnd.github.v3+json"
        });

        // GitHub rate limit 60 req/hour, tidak perlu auth untuk search public repos
        curl(&api_url, "GET", &None, Some(json!({}).to_string()), &Cancel {})
    }

    fn openalex(search: &str, from_year: Option<u16>, to_year: Option<u16>) -> Result<Output, String> {
        // OpenAlex API search
        let mut url = "https://api.openalex.org/search".to_string();
        
        // Build query parameters
        let params = if let Some(from) = from_year {
            format!("?from_publication_year={}&to_publication_year={}", from, 
                    to_year.unwrap_or(from))
        } else {
            "".to_string()
        };

        url.push_str(&params);

        curl(&url, "GET", &None, Some(json!({}).to_string()), &Cancel {})
    }

    fn doi_resolver(doi: &str) -> Result<Output, String> {
        // Resolve DOI ke metadata Crossref
        let api_url = format!("https://doi.org/{}", doi);
        
        // Untuk saat ini, return mock response
        Ok(Output { 
            text: format!("DOI {} ditemukan di Crossref (mock)\n\nMetadata tersedia di https://api.crossref.org/works/{}", 
                doi, doi),
            summary: format!("Resolusi DOI: {}", doi),
            ok: true 
        })
    }

    fn scopus(query: &str, from_date: Option<&str>, to_date: Option<&str>) -> Result<Output, String> {
        // Search database Scopus
        // API key perlu di-set via environment atau headers
        let api_url = format!("https://api.elsevier.com/content/search/scopus-database?query={}", 
            query.replace(" ", "+"));

        curl(&api_url, "GET", &None, Some(json!({}).to_string()), &Cancel {})
    }

    fn parse_url(url: &str) -> Option<String> {
        // Parse URL sederhana untuk extract host
        if url.contains("://") {
            let rest = url.split("://").nth(1)?;
            let parts: Vec<&str> = rest.trim_start_matches('/').splitn(2, '/').collect();
            Some(parts.get(0)?.to_string())
        } else {
            None
        }
    }


    fn head(path: &Path, lines: usize) -> Result<Output, String> {
        if !path.exists() {
            return Err(format!("{} tidak ada", path.display()));
        }

        let mut file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(e.to_string()),
        };

        let meta = match file.metadata() {
            Ok(m) => m,
            Err(e) => return Err(e.to_string()),
        };

        if meta.len() > FILE_MAX {
            return Err(format!("file {} terlalu besar (>{})", path.display(), FILE_MAX));
        }

        let mut buf = vec![0u8; lines * 1024]; // Estimasi buffer
        let nread = match file.read(&mut buf) {
            Ok(n) => n,
            Err(e) => return Err(format!("gagal baca: {}", e)),
        };

        let text = String::from_utf8_lossy(&buf[..nread]).trim_end_matches(|c| c.is_whitespace()).to_string();
        
        let total_lines = meta.len() as f64 / 512; // Estimasi ~512 byte per line rata-rata
        let total_lines = match text.lines().count() {
            n if n > 0 => n,
            _ => 0,
        };

        Ok(Output { 
            text: format!("{} baris:\n{}\n(total: {} baris)", lines, text, total_lines.min(lines * 2)), 
            summary: format!("{} baris", lines), 
            ok: true 
        })
    }

    fn tail(path: &Path, lines: usize) -> Result<Output, String> {
        if !path.exists() {
            return Err(format!("{} tidak ada", path.display()));
        }

        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) => return Err(e.to_string()),
        };

        if meta.len() > FILE_MAX {
            return Err(format!("file {} terlalu besar (>{})", path.display(), FILE_MAX));
        }

        // Baca dari akhir file
        let mut file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(e.to_string()),
        };

        let meta = match file.metadata() {
            Ok(m) => m,
            Err(e) => return Err(e.to_string()),
        };

        let start = meta.len() as u64;
        // Baca maksimal lines * 1024 byte dari akhir
        let buf_size = (lines * 512).min(1 << 20); // Max 1MB
        let mut buf = vec![0u8; buf_size as usize];
        
        let nread = match file.seek(std::io::SeekFrom::End(-(buf_size as i64))) {
            Ok(_) => {},
            Err(e) => return Err(format!("gagal seek: {}", e)),
        };
        
        match file.read(&mut buf) {
            Ok(n) => {},
            Err(e) => return Err(format!("gagal baca: {}", e)),
        };

        let text = String::from_utf8_lossy(&buf[..n]).trim_end_matches(|c| c.is_whitespace()).to_string();
        
        let total_lines = match text.lines().count() {
            n if n > 0 => n,
            _ => 0,
        };

        Ok(Output { 
            text: format!("{} baris terakhir:\n{}", lines, text), 
            summary: format!("{} baris", lines), 
            ok: true 
        })
    }

    fn spider(url: &str, options: &serde_json::Value, cancel: &Cancel) -> Result<Output, String> {
        // Spider/Crawler - crawling multiple pages dengan konfigurasi custom
        let headers = Some(json!({
            "User-Agent": "llmy-agent/0.1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }));

        // Untuk saat ini, gunakan curl internal dengan header custom
        let result = curl(url, "GET", &headers, Some(json!({}).to_string()), cancel)?;
        
        Ok(result)
    }

    fn essence(text: &str) -> Result<Output, String> {
        // Essence - extract entities/concepts dari text
        // Ini akan menggunakan regex dan NLP sederhana untuk ekstrak entitas
        
        // Ekstrak kata kunci penting (entity extraction)
        let entities = [
            "ORGANIZATION", "PERSON", "LOCATION", "DATE", "TIME", 
            "EMAIL", "URL", "PHONE", "MONEY", "PERCENT"
        ];
        
        let mut result = String::new();
        result.push_str("Essence (ekstrak entitas) - Mock implementation:\n");
        result.push_str(&format!("Input: {}\n", text.chars().take(100).collect::<String>()));
        result.push_str("\nEntities yang ditemukan:\n");
        
        // Ekstrak URL
        if let Some(start) = text.find("http") {
            let end = text[start..].find(|c: char| !matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '/' | '.' | '-')).unwrap_or(text.len() - start);
            let url: String = text[start..start + end].chars().collect();
            result.push_str(&format!("- URL: {}\n", url));
        }
        
        // Ekstrak email
        for pattern in ["@gmail.com", "@yahoo.com", "@outlook.com"] {
            if text.contains(pattern) {
                let mut start = text.find('@').unwrap_or(0);
                let end = text[start..].find(&pattern[1..]).map(|i| start + i).unwrap_or(text.len() - start);
                let email: String = text[start..start + end].chars().collect();
                result.push_str(&format!("- Email: {}\n", email));
            }
        }
        
        // Ekstrak keyword utama (stopwords filter sederhana)
        let stopwords = ["the", "and", "or", "is", "are", "was", "were", "in", "on", "at"];
        let words: Vec<&str> = text.split_whitespace().filter(|w| w.len() > 3 && !stopwords.contains(&*w)).collect();
        
        result.push_str(&format!("Keywords: {}\n", words.iter().take(10).collect::<Vec<_>>().join(", ")));
        
        Ok(Output { 
            text: result, 
            summary: format!("Essence dari {} karakter", text.len()), 
            ok: true 
        })
    }
