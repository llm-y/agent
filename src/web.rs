//! Tool internet: curl, github, openalex, doi, spider, dan (bila kuncinya ada di env) tavily,
//! exa, linkup, firecrawl, scopus, brightdata. Semuanya lewat `http.rs`, tanpa program luar,
//! dan selalu lewat izin: isi permintaan meninggalkan komputer ini.
//! Kunci hanya dibaca dari env dan hanya dikirim lewat https ke host layanannya.

use crate::cancel::Cancel;
use crate::http::{self, Req};
use crate::provider::env;
use crate::tools::Output;
use regex_lite::Regex;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const BODY_MAX: usize = 4 << 20;
const CUT: &str = "\u{0}cut";

/// Tool yang butuh kunci, dan nama variabel env-nya.
pub const KEYS: [(&str, &str); 6] = [
    ("tavily", "TAVILY_API_KEY"),
    ("exa", "EXA_API_KEY"),
    ("linkup", "LINKUP_API_KEY"),
    ("firecrawl", "FIRECRAWL_API_KEY"),
    ("scopus", "SCOPUS_API_KEY"),
    ("brightdata", "BRIGHTDATA_API_KEY"),
];

fn key_for(tool: &str) -> Option<&'static str> {
    KEYS.iter().find(|(t, _)| *t == tool).map(|(_, k)| *k)
}

fn tool(name: &str, desc: &str, props: Value, req: &[&str]) -> Value {
    json!({"type":"function","function":{"name":name,"description":desc,
        "parameters":{"type":"object","properties":props,"required":req}}})
}

fn s(d: &str) -> Value {
    json!({"type":"string","description":d})
}

fn i(d: &str) -> Value {
    json!({"type":"integer","description":d})
}

/// Daftar tool. Yang butuh kunci hanya muncul bila kuncinya ada, jadi model kecil tidak
/// menebak tool yang pasti gagal. Env dibaca sekali saat mulai, jadi daftar ini tetap
/// selama percakapan.
pub fn schema() -> Vec<Value> {
    let mut v = vec![
        tool("curl", "HTTP request to a URL. Returns the response body; HTML is converted to plain text unless raw is true. Follows redirects.",
             json!({"url": s("Full URL with http:// or https://"), "method": s("GET (default), POST, PUT, PATCH, DELETE"),
                    "headers": s("Optional request headers, one 'Name: value' per line"), "body": s("Optional request body"),
                    "raw": {"type":"boolean","description":"Return HTML as is"}}), &["url"]),
        tool("github", "GitHub REST API (GET, JSON). Use this instead of fetching github.com pages. Examples of path: search/repositories?q=rust+cli, repos/OWNER/REPO, repos/OWNER/REPO/readme, repos/OWNER/REPO/contents/PATH, repos/OWNER/REPO/issues, repos/OWNER/REPO/pulls, repos/OWNER/REPO/releases, repos/OWNER/REPO/contributors, users/NAME.",
             json!({"path": s("API path after https://api.github.com/")}), &["path"]),
        tool("openalex", "Search scholarly works in OpenAlex: title, year, authors, venue, DOI, citations, abstract.",
             json!({"search": s("Keywords"), "from_year": i("Optional first publication year"), "to_year": i("Optional last publication year"), "limit": i("Results, default 10, max 50")}), &["search"]),
        tool("doi", "Look up a DOI on Crossref: title, authors, journal, year, citations, abstract.",
             json!({"doi": s("DOI, e.g. 10.1038/nature14539")}), &["doi"]),
        tool("spider", "Crawl a website: fetch the start page and follow links on the same host. Returns the text of each page.",
             json!({"url": s("Start URL"), "depth": i("Link depth, default 1, max 3"), "limit": i("Max pages, default 5, max 30")}), &["url"]),
    ];
    let has = |t: &str| key_for(t).and_then(env).is_some();
    if has("tavily") {
        v.push(tool("tavily", "Web search for current information. Returns a short answer and sources with excerpts.",
            json!({"query": s("Search query"), "topic": {"type":"string","enum":["general","news"]}, "days": i("For news: how many days back"), "max_results": i("Default 5, max 20")}), &["query"]));
    }
    if has("exa") {
        v.push(tool("exa", "Semantic web search: finds pages by meaning. Returns title, URL, date, and text.",
            json!({"query": s("What to find, in natural language"), "num_results": i("Default 5, max 25"),
                   "category": s("Optional: company, research paper, news, pdf, github, tweet, personal site, financial report")}), &["query"]));
    }
    if has("linkup") {
        v.push(tool("linkup", "Web search with sourced results.",
            json!({"query": s("Search query"), "depth": {"type":"string","enum":["standard","deep"]}}), &["query"]));
    }
    if has("firecrawl") {
        v.push(tool("firecrawl", "Firecrawl: scrape one page to clean markdown, or crawl a site.",
            json!({"url": s("URL"), "action": {"type":"string","enum":["scrape","crawl"]}, "limit": i("Crawl: max pages, default 5, max 30")}), &["url"]));
    }
    if has("scopus") {
        v.push(tool("scopus", "Search Scopus (Elsevier) for articles with citation counts. Scopus syntax, e.g. TITLE-ABS-KEY(deep learning) AND PUBYEAR > 2020.",
            json!({"query": s("Scopus query"), "limit": i("Default 10, max 25")}), &["query"]));
    }
    if has("brightdata") {
        v.push(tool("brightdata", "Fetch a page through Bright Data Web Unlocker, for sites that block bots. Returns page text.",
            json!({"url": s("URL")}), &["url"]));
    }
    v
}

pub fn essence_schema() -> Value {
    tool("essence", "Extract the essentials from text without any network access: URLs, emails, DOIs, and top keywords.",
         json!({"text": s("Text to analyse")}), &["text"])
}

fn str_of(a: &Value, k: &str) -> Option<String> {
    match &a[k] {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn need(a: &Value, k: &str) -> Result<String, String> {
    str_of(a, k).ok_or(format!("argumen `{k}` wajib diisi"))
}

fn int_of(a: &Value, k: &str) -> Option<u64> {
    let v = &a[k];
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).or_else(|| v.as_str()?.trim().parse().ok())
}

fn bool_of(a: &Value, k: &str) -> bool {
    a[k].as_bool().unwrap_or(false) || a[k] == "true"
}

/// Validasi tanpa efek samping. None bila `name` bukan tool internet.
/// Hasilnya: judul, baris tampilan untuk persetujuan, dan argumen yang sudah dirapikan.
/// Judul, baris untuk layar izin, dan argumen yang sudah dirapikan.
pub type Call = (String, Vec<String>, Value);

pub fn prepare(name: &str, a: &Value) -> Option<Result<Call, String>> {
    let known = ["curl", "github", "openalex", "doi", "spider"].contains(&name) || key_for(name).is_some();
    if !known {
        return None;
    }
    Some(prepare_inner(name, a))
}

fn prepare_inner(name: &str, a: &Value) -> Result<Call, String> {
    if let Some(k) = key_for(name)
        && env(k).is_none()
    {
        return Err(format!("{name} butuh kunci di env {k}; tidak tersedia"));
    }
    if name == "brightdata" && env("BRIGHTDATA_ZONE").is_none() {
        return Err("brightdata butuh env BRIGHTDATA_ZONE (nama zone Web Unlocker)".into());
    }
    let mut lines = vec![];
    let (title, args) = match name {
        "curl" => {
            let url = need(a, "url")?;
            split_url(&url)?;
            let method = str_of(a, "method").unwrap_or("GET".into()).to_ascii_uppercase();
            if !["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"].contains(&method.as_str()) {
                return Err(format!("method {method} tidak didukung"));
            }
            let headers = parse_headers(&a["headers"])?;
            let body = str_of(a, "body").unwrap_or_default();
            for (k, v) in &headers {
                let secret = ["auth", "key", "token", "cookie", "secret"].iter().any(|w| k.to_ascii_lowercase().contains(w));
                lines.push(format!("{k}: {}", if secret { mask(v) } else { v.clone() }));
            }
            let n = body.lines().count();
            lines.extend(body.lines().take(12).map(str::to_string));
            if n > 12 {
                lines.push(format!("... {} baris lagi", n - 12));
            }
            let h: serde_json::Map<String, Value> = headers.into_iter().map(|(k, v)| (k, json!(v))).collect();
            (format!("curl {method} {url}"), json!({"url": url, "method": method, "headers": h, "body": body, "raw": bool_of(a, "raw")}))
        }
        "github" => {
            let path = github_path(&need(a, "path")?)?;
            (format!("github {path}"), json!({"path": path}))
        }
        "openalex" => {
            let q = need(a, "search")?;
            let mut v = json!({"search": q, "limit": int_of(a, "limit").unwrap_or(10).clamp(1, 50)});
            for k in ["from_year", "to_year"] {
                if let Some(y) = int_of(a, k).filter(|y| (1000..=3000).contains(y)) {
                    v[k] = json!(y);
                }
            }
            (format!("openalex {q}"), v)
        }
        "doi" => {
            let d = clean_doi(&need(a, "doi")?)?;
            (format!("doi {d}"), json!({"doi": d}))
        }
        "spider" => {
            let url = need(a, "url")?;
            split_url(&url)?;
            let depth = int_of(a, "depth").unwrap_or(1).min(3);
            let limit = int_of(a, "limit").unwrap_or(5).clamp(1, 30);
            (format!("spider {url} (kedalaman {depth}, maks. {limit} halaman)"), json!({"url": url, "depth": depth, "limit": limit}))
        }
        "tavily" => {
            let q = need(a, "query")?;
            let topic = if str_of(a, "topic").as_deref() == Some("news") { "news" } else { "general" };
            let mut v = json!({"query": q, "topic": topic, "max_results": int_of(a, "max_results").unwrap_or(5).clamp(1, 20)});
            if let Some(d) = int_of(a, "days").filter(|_| topic == "news") {
                v["days"] = json!(d.clamp(1, 365));
            }
            (format!("tavily {q}"), v)
        }
        "exa" => {
            let q = need(a, "query")?;
            let mut v = json!({"query": q, "num_results": int_of(a, "num_results").unwrap_or(5).clamp(1, 25)});
            if let Some(c) = str_of(a, "category") {
                v["category"] = json!(c);
            }
            (format!("exa {q}"), v)
        }
        "linkup" => {
            let q = need(a, "query")?;
            let depth = if str_of(a, "depth").as_deref() == Some("deep") { "deep" } else { "standard" };
            (format!("linkup {q}"), json!({"query": q, "depth": depth}))
        }
        "firecrawl" => {
            let url = need(a, "url")?;
            split_url(&url)?;
            let action = if str_of(a, "action").as_deref() == Some("crawl") { "crawl" } else { "scrape" };
            let limit = int_of(a, "limit").unwrap_or(5).clamp(1, 30);
            (format!("firecrawl {action} {url}"), json!({"url": url, "action": action, "limit": limit}))
        }
        "scopus" => {
            let q = need(a, "query")?;
            (format!("scopus {q}"), json!({"query": q, "limit": int_of(a, "limit").unwrap_or(10).clamp(1, 25)}))
        }
        "brightdata" => {
            let url = need(a, "url")?;
            split_url(&url)?;
            (format!("brightdata {url}"), json!({"url": url}))
        }
        _ => unreachable!(),
    };
    Ok((title, lines, args))
}

fn mask(v: &str) -> String {
    let n = v.chars().count();
    if n <= 8 { "***".into() } else { format!("{}...{}", v.chars().take(4).collect::<String>(), v.chars().skip(n - 4).collect::<String>()) }
}

/// Header dari teks `Nama: nilai` per baris, objek JSON, atau string berisi objek JSON.
fn parse_headers(v: &Value) -> Result<Vec<(String, String)>, String> {
    let obj = match v {
        Value::Null => return Ok(vec![]),
        Value::Object(o) => Some(o.clone()),
        Value::String(s) if s.trim().starts_with('{') => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(o)) => Some(o),
            _ => return Err("headers bukan JSON yang sah".into()),
        },
        Value::String(_) => None,
        _ => return Err("headers harus teks `Nama: nilai` per baris".into()),
    };
    let mut out = vec![];
    match obj {
        Some(o) => {
            for (k, v) in o {
                out.push((k, v.as_str().map(str::to_string).unwrap_or(v.to_string())));
            }
        }
        None => {
            for l in v.as_str().unwrap_or("").lines().filter(|l| !l.trim().is_empty()) {
                let (k, val) = l.split_once(':').ok_or(format!("header tanpa `:`: {l}"))?;
                out.push((k.trim().to_string(), val.trim().to_string()));
            }
        }
    }
    for (k, v) in &out {
        if k.is_empty() || k.contains(|c: char| !c.is_ascii_graphic() || c == ':') || v.contains(|c: char| c.is_control()) {
            return Err(format!("header {k:?} tidak sah"));
        }
        if ["host", "content-length", "connection", "transfer-encoding"].contains(&k.to_ascii_lowercase().as_str()) {
            return Err(format!("header {k} diatur sendiri oleh llmy"));
        }
    }
    Ok(out)
}

fn github_path(p: &str) -> Result<String, String> {
    let mut p = p.trim().to_string();
    for pre in ["https://api.github.com/", "http://api.github.com/", "api.github.com/"] {
        if let Some(r) = p.strip_prefix(pre) {
            p = r.into();
        }
    }
    // URL halaman github.com diterjemahkan ke API, bukan diambil HTML-nya.
    for pre in ["https://github.com/", "http://github.com/", "github.com/"] {
        if let Some(r) = p.strip_prefix(pre) {
            let parts: Vec<&str> = r.trim_end_matches('/').split('/').collect();
            p = match parts.as_slice() {
                [o, r] => format!("repos/{o}/{r}"),
                [o, r, "issues", n, ..] | [o, r, "pull", n, ..] => format!("repos/{o}/{r}/issues/{n}"),
                [o, r, "issues"] => format!("repos/{o}/{r}/issues"),
                [o, r, "pulls"] => format!("repos/{o}/{r}/pulls"),
                [o, r, "releases", ..] => format!("repos/{o}/{r}/releases"),
                [o, r, "blob" | "tree", rf, rest @ ..] => format!("repos/{o}/{r}/contents/{}?ref={rf}", rest.join("/")),
                [u] => format!("users/{u}"),
                _ => return Err("URL github.com ini tidak dikenali; pakai path API, mis. repos/OWNER/REPO".into()),
            };
        }
    }
    let p = p.trim_start_matches('/').replace(' ', "+");
    if p.is_empty() || p.contains("..") || p.contains(|c: char| c.is_control() || c == '#') {
        return Err("path GitHub tidak valid".into());
    }
    Ok(p)
}

fn clean_doi(d: &str) -> Result<String, String> {
    let mut d = d.trim();
    for pre in ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi.org/", "doi:", "DOI:", "DOI "] {
        if let Some(r) = d.strip_prefix(pre) {
            d = r.trim();
        }
    }
    if !d.starts_with("10.") || !d.contains('/') || d.contains(char::is_whitespace) {
        return Err(format!("`{d}` bukan DOI (harus berbentuk 10.xxxx/...)"));
    }
    Ok(d.to_string())
}

// ---- HTTP ----

struct Url {
    base: String,
    path: String,
    host: String,
    tls: bool,
}

fn split_url(u: &str) -> Result<Url, String> {
    let u = u.trim();
    let (tls, rest) = if let Some(r) = u.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = u.strip_prefix("http://") {
        (false, r)
    } else {
        return Err("URL harus diawali http:// atau https://".into());
    };
    let rest = rest.split('#').next().unwrap_or("");
    let at = rest.find(['/', '?']).unwrap_or(rest.len());
    let (host, path) = rest.split_at(at);
    if host.is_empty() || host.contains(|c: char| c == '@' || c.is_whitespace() || c.is_control()) {
        return Err(format!("URL tidak valid: {u}"));
    }
    if path.contains(|c: char| c.is_control()) {
        return Err("URL mengandung karakter kontrol".into());
    }
    let path = match path {
        "" => "/".to_string(),
        p if p.starts_with('?') => format!("/{p}"),
        p => p.replace(' ', "%20"),
    };
    let scheme = if tls { "https" } else { "http" };
    Ok(Url { base: format!("{scheme}://{host}"), path, host: host.to_ascii_lowercase(), tls })
}

fn join(u: &Url, loc: &str) -> String {
    let loc = loc.trim();
    if loc.starts_with("http://") || loc.starts_with("https://") {
        loc.to_string()
    } else if let Some(r) = loc.strip_prefix("//") {
        format!("{}://{r}", if u.tls { "https" } else { "http" })
    } else if loc.starts_with('/') {
        format!("{}{loc}", u.base)
    } else if loc.starts_with('?') {
        let p = u.path.split('?').next().unwrap_or("/");
        format!("{}{p}{loc}", u.base)
    } else {
        let p = u.path.split('?').next().unwrap_or("/");
        let dir = &p[..p.rfind('/').map_or(0, |i| i + 1)];
        let dir = if dir.is_empty() { "/" } else { dir };
        format!("{}{dir}{loc}", u.base)
    }
}

struct Resp {
    body: String,
    cut: bool,
    url: String,
}

/// Header hanya ikut pengalihan ke host yang sama dan tidak turun ke http, karena bisa
/// berisi kunci.
fn fetch(method: &str, url: &str, headers: &[(&str, String)], body: &str, follow: bool, cancel: &Cancel) -> Result<Resp, String> {
    let (mut url, mut method, mut body) = (url.to_string(), method.to_string(), body.to_string());
    for _ in 0..6 {
        let u = split_url(&url)?;
        let mut raw = String::new();
        let req = Req { base: &u.base, method: &method, path: &u.path, headers, body: &body };
        let r = http::request(&req, cancel, &mut |l| {
            if raw.len() + l.len() > BODY_MAX {
                return Err(CUT.into());
            }
            raw.push_str(l);
            raw.push('\n');
            Ok(())
        });
        match r {
            Ok(()) => return Ok(Resp { body: raw, cut: false, url }),
            Err(f) if f.msg == CUT => return Ok(Resp { body: raw, cut: true, url }),
            Err(f) if follow && (300..400).contains(&f.status) && f.location.is_some() => {
                let next = join(&u, f.location.as_deref().unwrap_or(""));
                let n = split_url(&next)?;
                if !headers.is_empty() && (n.host != u.host || (u.tls && !n.tls)) {
                    return Err(format!("dialihkan ke {next}; tidak diikuti karena membawa header"));
                }
                if f.status == 303 || (matches!(f.status, 301 | 302) && method == "POST") {
                    method = "GET".into();
                    body.clear();
                }
                url = next;
            }
            Err(f) if (300..400).contains(&f.status) => {
                return Err(format!("HTTP {}: dialihkan ke {}", f.status, f.location.unwrap_or_default()));
            }
            Err(f) => return Err(f.msg),
        }
    }
    Err("terlalu banyak pengalihan".into())
}

fn get_json(url: &str, headers: &[(&str, String)], cancel: &Cancel) -> Result<Value, String> {
    let r = fetch("GET", url, headers, "", false, cancel)?;
    serde_json::from_str(&r.body).map_err(|e| format!("balasan bukan JSON: {e}"))
}

fn post_json(url: &str, headers: &[(&str, String)], body: &Value, cancel: &Cancel) -> Result<Value, String> {
    let r = fetch("POST", url, headers, &body.to_string(), false, cancel)?;
    serde_json::from_str(&r.body).map_err(|e| format!("balasan bukan JSON: {e}"))
}

fn enc(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => o.push(b as char),
            b' ' => o.push('+'),
            _ => o += &format!("%{b:02X}"),
        }
    }
    o
}

fn key(tool: &str) -> Result<String, String> {
    let k = key_for(tool).unwrap_or("");
    env(k).ok_or(format!("env {k} belum diset"))
}

// ---- teks ----

fn re(cell: &'static OnceLock<Regex>, p: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(p).unwrap())
}

/// HTML ke teks biasa: buang skrip, gaya, dan tag; pertahankan pemisah blok.
pub fn html_text(h: &str) -> String {
    static DROP: OnceLock<Vec<Regex>> = OnceLock::new();
    static COMMENT: OnceLock<Regex> = OnceLock::new();
    static BLOCK: OnceLock<Regex> = OnceLock::new();
    static TAG: OnceLock<Regex> = OnceLock::new();
    let drop = DROP.get_or_init(|| {
        ["script", "style", "noscript", "svg", "template", "head"]
            .iter()
            .map(|t| Regex::new(&format!(r"(?is)<{t}\b.*?</{t}\s*>")).unwrap())
            .collect()
    });
    let title = page_title(h);
    let mut s = re(&COMMENT, r"(?s)<!--.*?-->").replace_all(h, "").into_owned();
    for r in drop {
        s = r.replace_all(&s, "\n").into_owned();
    }
    s = re(&BLOCK, r"(?i)<(/?(p|div|br|li|tr|h[1-6]|section|article|header|footer|nav|main|ul|ol|table|blockquote|pre|hr)\b[^>]*)>").replace_all(&s, "\n").into_owned();
    s = re(&TAG, r"(?s)<[^>]*>").replace_all(&s, "").into_owned();
    let s = entities(&s);
    let mut out = String::new();
    if let Some(t) = title {
        out += &format!("# {t}\n");
    }
    let mut blank = false;
    for l in s.lines() {
        let l = l.split_whitespace().collect::<Vec<_>>().join(" ");
        if l.is_empty() {
            blank = true;
            continue;
        }
        if blank && !out.is_empty() {
            out.push('\n');
        }
        blank = false;
        out += &l;
        out.push('\n');
    }
    out
}

fn page_title(h: &str) -> Option<String> {
    static T: OnceLock<Regex> = OnceLock::new();
    let c = re(&T, r"(?is)<title[^>]*>(.*?)</title>").captures(h)?;
    let t = entities(c.get(1)?.as_str()).split_whitespace().collect::<Vec<_>>().join(" ");
    (!t.is_empty()).then_some(t)
}

fn entities(s: &str) -> String {
    static NUM: OnceLock<Regex> = OnceLock::new();
    let s = re(&NUM, r"&#(x[0-9a-fA-F]+|[0-9]+);").replace_all(s, |c: &regex_lite::Captures| {
        let n = &c[1];
        let v = match n.strip_prefix('x') {
            Some(h) => u32::from_str_radix(h, 16).ok(),
            None => n.parse().ok(),
        };
        v.and_then(char::from_u32).filter(|c| !c.is_control() || *c == '\n').map(String::from).unwrap_or_default()
    });
    s.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
        .replace("&apos;", "'").replace("&mdash;", "—").replace("&ndash;", "–").replace("&hellip;", "…")
        .replace("&amp;", "&")
}

fn looks_html(s: &str) -> bool {
    let h: String = s.chars().take(1024).collect::<String>().to_ascii_lowercase();
    h.contains("<!doctype html") || h.contains("<html") || h.contains("<body") || h.contains("<head")
}

fn cut(s: &str, max: usize) -> (String, bool) {
    if s.len() <= max {
        return (s.to_string(), false);
    }
    let mut i = max;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    (format!("{}\n... [{} byte dipotong]", &s[..i], s.len() - i), true)
}

fn done(text: String, summary: String, max: usize) -> Output {
    let (text, _) = cut(&text, max);
    Output { text, summary, ok: true }
}

fn kb(n: usize) -> String {
    if n < 1024 { format!("{n} B") } else { format!("{:.1} KB", n as f64 / 1024.0) }
}

/// Buang isian API yang tidak berguna bagi model (URL API, id node) agar hemat token.
fn prune(v: &mut Value) {
    match v {
        Value::Object(o) => {
            o.retain(|k, _| !((k.ends_with("_url") && k != "html_url") || k == "node_id" || k == "gravatar_id" || k == "_links"));
            for x in o.values_mut() {
                prune(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(prune),
        _ => {}
    }
}

fn b64decode(s: &str) -> Option<Vec<u8>> {
    let mut out = vec![];
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn j(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

// ---- jalankan ----

pub fn run(tool: &str, a: &Value, max: usize, cancel: &Cancel) -> Output {
    let r = match tool {
        "curl" => curl(a, max, cancel),
        "github" => github(a, max, cancel),
        "openalex" => openalex(a, max, cancel),
        "doi" => doi(a, max, cancel),
        "spider" => spider(a, max, cancel),
        "tavily" => tavily(a, max, cancel),
        "exa" => exa(a, max, cancel),
        "linkup" => linkup(a, max, cancel),
        "firecrawl" => firecrawl(a, max, cancel),
        "scopus" => scopus(a, max, cancel),
        "brightdata" => brightdata(a, max, cancel),
        _ => Err(format!("tool {tool} tidak ada")),
    };
    r.unwrap_or_else(|e| if cancel.is_set() { Output::err("dibatalkan pengguna") } else { Output::err(e) })
}

fn curl(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let url = j(&a["url"]);
    let headers: Vec<(String, String)> = a["headers"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), j(v))).collect()).unwrap_or_default();
    let h: Vec<(&str, String)> = headers.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let r = fetch(&j(&a["method"]), &url, &h, a["body"].as_str().unwrap_or(""), true, cancel)?;
    let html = looks_html(&r.body) && !bool_of(a, "raw");
    let mut text = if html { html_text(&r.body) } else { r.body.clone() };
    if r.url != url {
        text = format!("(dialihkan ke {})\n{text}", r.url);
    }
    if r.cut {
        text += "\n... [balasan terlalu besar, dipotong]";
    }
    let summary = format!("{}{}", kb(r.body.len()), if html { ", HTML jadi teks" } else { "" });
    Ok(done(if text.trim().is_empty() { "(balasan kosong)".into() } else { text }, summary, max))
}

fn github(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let path = j(&a["path"]);
    let mut h: Vec<(&str, String)> = vec![("Accept", "application/vnd.github+json".into()), ("X-GitHub-Api-Version", "2022-11-28".into())];
    if let Some(t) = env("GITHUB_TOKEN") {
        h.push(("Authorization", format!("Bearer {t}")));
    }
    let mut v = get_json(&format!("https://api.github.com/{path}"), &h, cancel)?;
    // readme dan contents: isi file dikirim base64.
    if v["encoding"] == "base64"
        && let Some(c) = v["content"].as_str()
        && let Some(b) = b64decode(c)
    {
        let name = j(&v["path"]);
        let text = String::from_utf8_lossy(&b).into_owned();
        return Ok(done(format!("{name}\n\n{text}"), format!("{name}, {}", kb(b.len())), max));
    }
    prune(&mut v);
    let summary = match &v {
        Value::Array(x) => format!("{} entri", x.len()),
        o if o["total_count"].is_number() => format!("{} hasil", o["total_count"]),
        _ => "ok".into(),
    };
    Ok(done(serde_json::to_string_pretty(&v).unwrap_or_default(), summary, max))
}

fn openalex(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let mut url = format!("https://api.openalex.org/works?search={}&per-page={}", enc(&j(&a["search"])), a["limit"].as_u64().unwrap_or(10));
    let mut f = vec![];
    if let Some(y) = a["from_year"].as_u64() {
        f.push(format!("from_publication_date:{y}-01-01"));
    }
    if let Some(y) = a["to_year"].as_u64() {
        f.push(format!("to_publication_date:{y}-12-31"));
    }
    if !f.is_empty() {
        url += &format!("&filter={}", f.join(","));
    }
    let v = get_json(&url, &[], cancel)?;
    let items = v["results"].as_array().cloned().unwrap_or_default();
    let mut out = format!("{} hasil total\n", v["meta"]["count"]);
    for w in &items {
        let authors: Vec<String> = w["authorships"].as_array().map(|x| x.iter().take(6).map(|a| j(&a["author"]["display_name"])).collect()).unwrap_or_default();
        out += &format!("\n- {} ({})\n  {}\n", j(&w["title"]), j(&w["publication_year"]), authors.join(", "));
        let venue = j(&w["primary_location"]["source"]["display_name"]);
        if !venue.is_empty() {
            out += &format!("  {venue}\n");
        }
        out += &format!("  DOI: {} · sitasi: {} · {}\n", j(&w["doi"]), j(&w["cited_by_count"]), j(&w["id"]));
        let abs = abstract_of(&w["abstract_inverted_index"]);
        if !abs.is_empty() {
            out += &format!("  Abstrak: {}\n", abs.chars().take(600).collect::<String>());
        }
    }
    Ok(done(out, format!("{} hasil", items.len()), max))
}

/// OpenAlex menyimpan abstrak sebagai indeks kata -> posisi.
fn abstract_of(v: &Value) -> String {
    let Some(o) = v.as_object() else { return String::new() };
    let mut w: Vec<(u64, &str)> = vec![];
    for (k, pos) in o {
        for p in pos.as_array().into_iter().flatten() {
            if let Some(p) = p.as_u64() {
                w.push((p, k));
            }
        }
    }
    w.sort();
    w.iter().map(|x| x.1).collect::<Vec<_>>().join(" ")
}

fn doi(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let d = j(&a["doi"]);
    let path: String = d.chars().map(|c| match c { ' ' => "%20".into(), '#' => "%23".into(), '?' => "%3F".into(), c => c.to_string() }).collect();
    let v = get_json(&format!("https://api.crossref.org/works/{path}"), &[], cancel)?;
    let m = &v["message"];
    let first = |k: &str| j(&m[k][0]);
    let authors: Vec<String> = m["author"].as_array().map(|x| x.iter().map(|a| format!("{} {}", j(&a["given"]), j(&a["family"])).trim().to_string()).collect()).unwrap_or_default();
    let year = j(&m["issued"]["date-parts"][0][0]);
    let mut out = format!("{}\n{}\n{} ({year})\n", first("title"), authors.join(", "), first("container-title"));
    out += &format!("Jenis: {} · Penerbit: {} · Sitasi: {}\nDOI: {} · {}\n", j(&m["type"]), j(&m["publisher"]), j(&m["is-referenced-by-count"]), j(&m["DOI"]), j(&m["URL"]));
    for (k, label) in [("volume", "Volume"), ("issue", "Nomor"), ("page", "Halaman")] {
        if !m[k].is_null() {
            out += &format!("{label}: {}\n", j(&m[k]));
        }
    }
    if let Some(ab) = m["abstract"].as_str() {
        out += &format!("\nAbstrak: {}\n", html_text(ab).trim());
    }
    Ok(done(out, first("title").chars().take(60).collect(), max))
}

fn spider(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    static HREF: OnceLock<Regex> = OnceLock::new();
    // Bentuk baku agar `https://x` dan `https://X/` tidak dikunjungi dua kali.
    let norm = |u: &Url| format!("{}://{}{}", if u.tls { "https" } else { "http" }, u.host, u.path);
    let first = split_url(&j(&a["url"]))?;
    let (start, host) = (norm(&first), first.host.clone());
    let depth = a["depth"].as_u64().unwrap_or(1);
    let limit = a["limit"].as_u64().unwrap_or(5) as usize;
    let per = (max / limit).max(500);
    let skip = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".gz", ".css", ".js", ".mp4", ".mp3", ".woff", ".woff2"];
    let (mut queue, mut seen) = (VecDeque::from([(start.clone(), 0u64)]), HashSet::from([start.clone()]));
    let (mut out, mut pages, mut fails) = (String::new(), 0, 0);
    while let Some((url, d)) = queue.pop_front() {
        if pages >= limit || cancel.is_set() {
            break;
        }
        pages += 1;
        match fetch("GET", &url, &[], "", true, cancel) {
            Ok(r) => {
                let text = if looks_html(&r.body) { html_text(&r.body) } else { r.body.clone() };
                out += &format!("\n=== {url}\n{}\n", cut(text.trim(), per).0);
                if d < depth {
                    let base = split_url(&r.url)?;
                    for c in re(&HREF, r#"(?i)<a\b[^>]*?\bhref\s*=\s*["']([^"'#]+)"#).captures_iter(&r.body) {
                        let href = entities(&c[1]);
                        if href.starts_with("mailto:") || href.starts_with("javascript:") || href.starts_with("tel:") {
                            continue;
                        }
                        let next = join(&base, &href);
                        let Ok(n) = split_url(&next) else { continue };
                        let next = norm(&n);
                        let lower = n.path.split('?').next().unwrap_or("").to_ascii_lowercase();
                        if n.host == host && !skip.iter().any(|e| lower.ends_with(e)) && seen.insert(next.clone()) {
                            queue.push_back((next, d + 1));
                        }
                    }
                }
            }
            Err(e) => {
                fails += 1;
                out += &format!("\n=== {url}\nError: {e}\n");
            }
        }
    }
    if pages == fails {
        return Err(out.trim().to_string());
    }
    let more = queue.len();
    if more > 0 {
        out += &format!("\n({more} tautan lain belum dikunjungi)\n");
    }
    Ok(done(out, format!("{pages} halaman{}", if fails > 0 { format!(", {fails} gagal") } else { String::new() }), max))
}

fn tavily(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let h = [("Authorization", format!("Bearer {}", key("tavily")?))];
    let mut body = json!({"query": a["query"], "topic": a["topic"], "max_results": a["max_results"], "include_answer": true});
    if !a["days"].is_null() {
        body["days"] = a["days"].clone();
    }
    let v = post_json("https://api.tavily.com/search", &h, &body, cancel)?;
    let mut out = String::new();
    if let Some(ans) = v["answer"].as_str() {
        out += &format!("Jawaban: {ans}\n");
    }
    let items = v["results"].as_array().cloned().unwrap_or_default();
    for r in &items {
        out += &format!("\n- {}\n  {}\n  {}\n", j(&r["title"]), j(&r["url"]), j(&r["content"]).trim());
    }
    Ok(done(out, format!("{} sumber", items.len()), max))
}

fn exa(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let h = [("x-api-key", key("exa")?)];
    let mut body = json!({"query": a["query"], "numResults": a["num_results"], "contents": {"text": {"maxCharacters": 1500}}});
    if !a["category"].is_null() {
        body["category"] = a["category"].clone();
    }
    let v = post_json("https://api.exa.ai/search", &h, &body, cancel)?;
    let items = v["results"].as_array().cloned().unwrap_or_default();
    let mut out = String::new();
    for r in &items {
        out += &format!("\n- {}\n  {} {}\n  {}\n", j(&r["title"]), j(&r["url"]), j(&r["publishedDate"]), j(&r["text"]).trim());
    }
    Ok(done(out, format!("{} hasil", items.len()), max))
}

fn linkup(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let h = [("Authorization", format!("Bearer {}", key("linkup")?))];
    let body = json!({"q": a["query"], "depth": a["depth"], "outputType": "searchResults"});
    let v = post_json("https://api.linkup.so/v1/search", &h, &body, cancel)?;
    let items = v["results"].as_array().cloned().unwrap_or_default();
    let mut out = String::new();
    for r in &items {
        out += &format!("\n- {}\n  {}\n  {}\n", j(&r["name"]), j(&r["url"]), j(&r["content"]).trim());
    }
    Ok(done(out, format!("{} hasil", items.len()), max))
}

fn firecrawl(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let h = [("Authorization", format!("Bearer {}", key("firecrawl")?))];
    let page = |d: &Value| format!("=== {}\n{}\n", j(&d["metadata"]["sourceURL"]), j(&d["markdown"]).trim());
    if a["action"] == "scrape" {
        let v = post_json("https://api.firecrawl.dev/v1/scrape", &h, &json!({"url": a["url"], "formats": ["markdown"], "onlyMainContent": true}), cancel)?;
        let md = j(&v["data"]["markdown"]);
        return Ok(done(page(&v["data"]), kb(md.len()), max));
    }
    let v = post_json("https://api.firecrawl.dev/v1/crawl", &h, &json!({"url": a["url"], "limit": a["limit"], "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": true}}), cancel)?;
    let id = v["id"].as_str().ok_or(format!("firecrawl tidak memberi id: {v}"))?.to_string();
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("id crawl tidak valid".into());
    }
    let start = Instant::now();
    loop {
        for _ in 0..20 {
            if cancel.is_set() {
                return Err("dibatalkan".into());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let s = get_json(&format!("https://api.firecrawl.dev/v1/crawl/{id}"), &h, cancel)?;
        let status = j(&s["status"]);
        if status == "completed" || start.elapsed() > Duration::from_secs(180) {
            let data = s["data"].as_array().cloned().unwrap_or_default();
            let mut out: String = data.iter().map(page).collect();
            if status != "completed" {
                out = format!("(crawl belum selesai setelah 3 menit; status {status}; id {id})\n{out}");
            }
            return Ok(done(out, format!("{} halaman", data.len()), max));
        }
        if status == "failed" || status == "cancelled" {
            return Err(format!("crawl {status}"));
        }
    }
}

fn scopus(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let h = [("X-ELS-APIKey", key("scopus")?), ("Accept", "application/json".into())];
    let url = format!("https://api.elsevier.com/content/search/scopus?query={}&count={}", enc(&j(&a["query"])), a["limit"].as_u64().unwrap_or(10));
    let v = get_json(&url, &h, cancel)?;
    let r = &v["search-results"];
    let items: Vec<Value> = r["entry"].as_array().cloned().unwrap_or_default().into_iter().filter(|e| e["error"].is_null()).collect();
    let mut out = format!("{} hasil total\n", j(&r["opensearch:totalResults"]));
    for e in &items {
        out += &format!("\n- {}\n  {} · {} ({})\n  DOI: {} · sitasi: {}\n", j(&e["dc:title"]), j(&e["dc:creator"]), j(&e["prism:publicationName"]),
            j(&e["prism:coverDate"]), j(&e["prism:doi"]), j(&e["citedby-count"]));
    }
    Ok(done(out, format!("{} hasil", items.len()), max))
}

fn brightdata(a: &Value, max: usize, cancel: &Cancel) -> Result<Output, String> {
    let h = [("Authorization", format!("Bearer {}", key("brightdata")?))];
    let zone = env("BRIGHTDATA_ZONE").ok_or("env BRIGHTDATA_ZONE belum diset")?;
    let body = json!({"zone": zone, "url": a["url"], "format": "raw"});
    let r = fetch("POST", "https://api.brightdata.com/request", &h, &body.to_string(), false, cancel)?;
    let text = if looks_html(&r.body) { html_text(&r.body) } else { r.body.clone() };
    Ok(done(text, kb(r.body.len()), max))
}

// ---- essence: tanpa jaringan ----

const STOP: &str = "yang dan di ke dari untuk dengan pada ini itu dalam adalah atau juga tidak akan oleh sebagai karena bisa dapat ada lebih sudah telah saat secara kami kita mereka anda saya para serta bagi agar namun hanya maka jika tersebut antara setelah sangat banyak the and for with that this from are was were have has had not but you your they their them our its can will would should could about into than then there what which when where who how also more most such only other some any each been being over under after before between through these those very just out all may use used using does did one two new";

pub fn essence(text: &str) -> Output {
    static URL: OnceLock<Regex> = OnceLock::new();
    static MAIL: OnceLock<Regex> = OnceLock::new();
    static DOI: OnceLock<Regex> = OnceLock::new();
    let uniq = |it: Vec<String>| {
        let mut seen = HashSet::new();
        it.into_iter().filter(|x| seen.insert(x.clone())).collect::<Vec<_>>()
    };
    let urls = uniq(re(&URL, r#"https?://[^\s<>"'\)\]]+"#).find_iter(text).map(|m| m.as_str().trim_end_matches(['.', ',', ';']).to_string()).collect());
    let mails = uniq(re(&MAIL, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").find_iter(text).map(|m| m.as_str().to_string()).collect());
    let dois = uniq(re(&DOI, r#"\b10\.\d{4,9}/[^\s"<>]+"#).find_iter(text).map(|m| m.as_str().trim_end_matches(['.', ',', ';', ')']).to_string()).collect());
    let stop: HashSet<&str> = STOP.split_whitespace().collect();
    let mut freq: HashMap<String, usize> = HashMap::new();
    let clean = re(&URL, r#"https?://[^\s<>"'\)\]]+"#).replace_all(text, " ");
    let clean = re(&DOI, r#"\b10\.\d{4,9}/[^\s"<>]+"#).replace_all(&clean, " ");
    for w in clean.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_')) {
        let w = w.trim_matches(['-', '_']).to_lowercase();
        if w.chars().count() < 4 || !w.chars().next().is_some_and(char::is_alphabetic) || mails.iter().any(|m| m.to_lowercase().contains(&w)) {
            continue;
        }
        if !stop.contains(w.as_str()) {
            *freq.entry(w).or_default() += 1;
        }
    }
    let mut top: Vec<(String, usize)> = freq.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(20);
    let words = text.split_whitespace().count();
    let mut out = format!("{words} kata, {} karakter\n", text.chars().count());
    for (label, v) in [("URL", &urls), ("Email", &mails), ("DOI", &dois)] {
        if !v.is_empty() {
            out += &format!("\n{label} ({}):\n- {}\n", v.len(), v.join("\n- "));
        }
    }
    if !top.is_empty() {
        out += &format!("\nKata kunci: {}\n", top.iter().map(|(w, n)| format!("{w} ({n})")).collect::<Vec<_>>().join(", "));
    }
    let summary = format!("{} URL, {} email, {} DOI, {} kata kunci", urls.len(), mails.len(), dois.len(), top.len());
    Output { text: out, summary, ok: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_dipecah_dan_digabung() {
        let u = split_url("https://Example.com/a/b?x=1#frag").unwrap();
        assert_eq!((u.base.as_str(), u.path.as_str(), u.host.as_str()), ("https://Example.com", "/a/b?x=1", "example.com"));
        assert_eq!(split_url("http://h").unwrap().path, "/");
        assert_eq!(split_url("http://h?q=1").unwrap().path, "/?q=1");
        assert!(split_url("ftp://h").is_err() && split_url("https://user@h/").is_err() && split_url("h.com").is_err());
        assert_eq!(join(&u, "c"), "https://Example.com/a/c");
        assert_eq!(join(&u, "/d"), "https://Example.com/d");
        assert_eq!(join(&u, "//cdn.x/y"), "https://cdn.x/y");
        assert_eq!(join(&u, "http://z/"), "http://z/");
        assert_eq!(join(&u, "?p=2"), "https://Example.com/a/b?p=2");
    }

    #[test]
    fn html_jadi_teks() {
        let h = "<html><head><title>Judul &amp; Sub</title><style>x{}</style></head><body><script>alert(1)</script>\
                 <h1>Halo</h1><p>satu&nbsp;dua <b>tiga</b></p><!-- rahasia --><div>&#65;&#x42;</div></body></html>";
        let t = html_text(h);
        assert!(t.starts_with("# Judul & Sub\n"), "{t}");
        assert!(t.contains("Halo") && t.contains("satu dua tiga") && t.contains("AB"), "{t}");
        assert!(!t.contains("alert") && !t.contains("rahasia") && !t.contains('<') && !t.contains("x{}"), "{t}");
    }

    #[test]
    fn header_divalidasi() {
        let h = parse_headers(&json!("Accept: text/plain\nX-A: 1")).unwrap();
        assert_eq!(h, vec![("Accept".into(), "text/plain".into()), ("X-A".into(), "1".into())]);
        assert_eq!(parse_headers(&json!({"A": "b"})).unwrap(), vec![("A".into(), "b".into())]);
        assert_eq!(parse_headers(&json!("{\"A\": \"b\"}")).unwrap(), vec![("A".into(), "b".into())]);
        assert!(parse_headers(&json!("Host: x")).is_err());
        assert!(parse_headers(&json!("tanpa titik dua")).is_err());
        assert!(parse_headers(&json!({"A\r\nB": "x"})).is_err());
    }

    #[test]
    fn github_dari_url_halaman() {
        assert_eq!(github_path("https://github.com/llm-y/agent").unwrap(), "repos/llm-y/agent");
        assert_eq!(github_path("github.com/a/b/blob/main/src/x.rs").unwrap(), "repos/a/b/contents/src/x.rs?ref=main");
        assert_eq!(github_path("/search/repositories?q=rust cli").unwrap(), "search/repositories?q=rust+cli");
        assert_eq!(github_path("https://api.github.com/users/x").unwrap(), "users/x");
        assert!(github_path("repos/../x").is_err());
    }

    #[test]
    fn doi_dirapikan() {
        assert_eq!(clean_doi("https://doi.org/10.1038/nature14539").unwrap(), "10.1038/nature14539");
        assert_eq!(clean_doi("doi: 10.1/x").unwrap(), "10.1/x");
        assert!(clean_doi("bukan doi").is_err());
    }

    #[test]
    fn kunci_wajib_untuk_tool_berkunci() {
        // Env uji tidak boleh punya kunci asli; lewati bila ada.
        if env("TAVILY_API_KEY").is_some() {
            return;
        }
        assert!(prepare("tavily", &json!({"query": "x"})).unwrap().is_err());
        assert!(!schema().iter().any(|t| t["function"]["name"] == "tavily"));
        assert!(prepare("read", &json!({})).is_none());
        let (title, lines, args) = prepare("curl", &json!({"url": "https://x.y/z", "headers": "Authorization: Bearer abcdefghijkl"})).unwrap().unwrap();
        assert_eq!(title, "curl GET https://x.y/z");
        assert_eq!(lines, vec!["Authorization: Bear...ijkl"]);
        assert_eq!(args["headers"]["Authorization"], "Bearer abcdefghijkl");
    }

    #[test]
    fn essence_mengambil_inti() {
        let o = essence("Lihat https://ll.my.id/docs, kirim ke a.b@c.id. DOI 10.1038/nature14539. agent agent agent rust rust");
        assert!(o.text.contains("https://ll.my.id/docs\n") && o.text.contains("a.b@c.id") && o.text.contains("10.1038/nature14539"), "{}", o.text);
        assert!(o.text.contains("agent (3), rust (2)") && !o.text.split("Kata kunci").nth(1).unwrap_or("").contains("nature"), "{}", o.text);
    }

    /// Server satu kali: balas `resp` dan kembalikan request yang diterima.
    fn serve(resp: String) -> (u16, std::thread::JoinHandle<String>) {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut b = [0u8; 4096];
            let n = c.read(&mut b).unwrap();
            c.write_all(resp.as_bytes()).unwrap();
            String::from_utf8_lossy(&b[..n]).into_owned()
        });
        (port, h)
    }

    #[test]
    fn pengalihan_tidak_membawa_header_ke_host_lain() {
        let c = Cancel::default();
        let (pb, hb) = serve("HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nsampai".into());
        let redirect = |to: String| format!("HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\n\r\n");
        // Tanpa header: diikuti ke host lain.
        let (pa, ha) = serve(redirect(format!("http://localhost:{pb}/b")));
        let r = fetch("GET", &format!("http://127.0.0.1:{pa}/a"), &[], "", true, &c).unwrap();
        assert_eq!((r.body.trim(), r.url.as_str()), ("sampai", format!("http://localhost:{pb}/b").as_str()));
        ha.join().unwrap();
        assert!(hb.join().unwrap().starts_with("GET /b "));
        // Dengan header (bisa berisi kunci): berhenti.
        let (pa, ha) = serve(redirect("http://localhost:1/x".into()));
        let e = fetch("GET", &format!("http://127.0.0.1:{pa}/a"), &[("Authorization", "Bearer k".into())], "", true, &c).err().unwrap();
        assert!(e.contains("tidak diikuti"), "{e}");
        assert!(ha.join().unwrap().contains("Authorization: Bearer k"));
    }

    #[test]
    fn base64_github() {
        assert_eq!(b64decode("aGFs\nbyBkdW5pYQ==").unwrap(), b"halo dunia");
    }

    #[test]
    fn prune_membuang_url_api() {
        let mut v = json!({"name":"a","html_url":"h","url":"u","issues_url":"x","node_id":"n","owner":{"login":"o","avatar_url":"z"}});
        prune(&mut v);
        assert_eq!(v, json!({"name":"a","html_url":"h","url":"u","owner":{"login":"o"}}));
    }
}
