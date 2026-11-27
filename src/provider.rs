//! Memilih dan memanggil model. Riwayat disimpan dalam satu bentuk netral (ala Ollama:
//! role system/user/assistant/tool); tiap provider menerjemahkannya sendiri.

use crate::anthropic;
use crate::api;
use crate::cancel::Cancel;
use crate::detect::{self, Tools};
use crate::gemini;
use crate::http::{self, Fail, Req};
use crate::ollama;
use crate::openai;
use crate::text;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Ollama,
    Anthropic,
    OpenAi,
    Gemini,
}

pub enum Delta<'a> {
    Text(&'a str),
    Think,
    /// Kegiatan selain menjawab, mis. menunggu sebelum mencoba lagi.
    Work(&'static str),
    Note(&'static str),
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Turn,
    /// Meringkas: model tidak boleh memanggil tool.
    Summary,
}

#[derive(Default)]
pub struct Reply {
    pub content: String,
    pub thinking: String,
    /// Thinking ala OpenAI (`reasoning_content`); dikirim balik hanya bila server memintanya.
    pub reasoning: String,
    pub calls: Vec<Value>,
    /// Blok asli Anthropic, harus dikirim balik persis sama.
    pub blocks: Option<Value>,
    /// Part asli Gemini (`thought` / `thoughtSignature`).
    pub parts: Option<Value>,
    pub tokens: u64,
    /// "length" = terpotong, "refusal" = ditolak, lainnya = selesai.
    pub done_reason: String,
    pub note: Option<String>,
    /// Thinking lama ditolak dan dibuang; riwayat pemanggil harus ikut dibersihkan.
    pub stripped: bool,
}

#[derive(Clone)]
pub struct Provider {
    pub kind: Kind,
    pub profile: String,
    pub base: String,
    pub model: String,
    pub ctx: u64,
    pub max_out: u64,
    pub think: bool,
    pub effort: Option<String>,
    pub tools: Tools,
    /// Tool lewat teks, untuk server yang tidak meneruskan `tools`.
    pub text_tools: bool,
    /// Server mewajibkan `reasoning_content` dikirim balik.
    pub echo: bool,
    /// Server tetap menolak riwayat thinking: matikan thinking.
    pub no_think: bool,
    pub detected: bool,
    /// Setelan per fase BDI: suhu, dan minta JSON (Ollama `format`, Gemini mime type).
    pub temperature: Option<f64>,
    pub json: bool,
    /// Status HTTP kegagalan terakhir (0 = sebelum ada status); untuk memilih cadangan.
    pub failed: Option<u16>,
    /// Pengaturan API khusus sesi aktif (tidak ditulis ke disk).
    pub session_apis: BTreeMap<String, api::Profile>,
    api_key: Option<String>,
    auth_bearer: bool,
    has_url: bool,
    ctx_user: Option<u64>,
    probed: bool,
}

/// Context cloud dibatasi agar biaya tidak melonjak; `--ctx` menaikkannya.
const CLOUD_CTX: u64 = 200_000;
const OLLAMA_CTX: u64 = 32_768;
const RETRIES: u32 = 2;

pub fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn env_url(kind: Kind) -> Option<String> {
    match kind {
        Kind::Ollama => env("LLMY_URL"),
        Kind::Anthropic => env("ANTHROPIC_BASE_URL"),
        Kind::OpenAi => env("OPENAI_BASE_URL"),
        Kind::Gemini => env("GEMINI_BASE_URL"),
    }
}

fn env_key(kind: Kind) -> (Option<String>, bool) {
    match kind {
        Kind::Ollama => (None, false),
        Kind::Anthropic => match (env("ANTHROPIC_API_KEY"), env("ANTHROPIC_AUTH_TOKEN")) {
            (Some(k), _) => (Some(k), false),
            (None, Some(t)) => (Some(t), true),
            _ => (None, false),
        },
        Kind::OpenAi => (env("OPENAI_API_KEY"), false),
        Kind::Gemini => (env("GEMINI_API_KEY").or_else(|| env("GOOGLE_API_KEY")), false),
    }
}

fn env_key_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Ollama => "",
        Kind::Anthropic => "ANTHROPIC_API_KEY",
        Kind::OpenAi => "OPENAI_API_KEY",
        Kind::Gemini => "GEMINI_API_KEY",
    }
}

/// "anthropic/x", "openai/x", "gemini/x", "ollama/x", atau "<profil>/x".
fn resolve_spec(spec: &str, cfg: &api::Config, sess: &BTreeMap<String, api::Profile>) -> Result<(String, Kind, String), String> {
    if let Some((prefix, model)) = spec.split_once('/') {
        if let Some(p) = sess.get(prefix).or_else(|| cfg.providers.get(prefix)) {
            return Ok((prefix.into(), p.protocol, model.into()));
        }
        if let Some((k, _)) = api::preset(prefix) {
            return Ok((prefix.into(), k, model.into()));
        }
        return Ok(("ollama".into(), Kind::Ollama, spec.into()));
    }
    if spec != "ollama"
        && let Some(p) = sess.get(spec).or_else(|| cfg.providers.get(spec))
    {
        let m = p.model.clone().ok_or_else(|| {
            format!("model bawaan untuk profil '{spec}' belum diatur; pakai `{spec}/<model>`")
        })?;
        return Ok((spec.into(), p.protocol, m));
    }
    if spec.starts_with("claude-") {
        return Ok(("anthropic".into(), Kind::Anthropic, spec.into()));
    }
    if spec.starts_with("gemini-") {
        return Ok(("gemini".into(), Kind::Gemini, spec.into()));
    }
    Ok(("ollama".into(), Kind::Ollama, spec.into()))
}

#[cfg(test)]
fn split(spec: &str) -> (Kind, &str) {
    for (p, k) in [
        ("anthropic/", Kind::Anthropic),
        ("openai/", Kind::OpenAi),
        ("gemini/", Kind::Gemini),
        ("ollama/", Kind::Ollama),
    ] {
        if let Some(m) = spec.strip_prefix(p) {
            return (k, m);
        }
    }
    if spec.starts_with("claude-") {
        (Kind::Anthropic, spec)
    } else if spec.starts_with("gemini-") {
        (Kind::Gemini, spec)
    } else {
        (Kind::Ollama, spec)
    }
}

impl Provider {
    /// `url` hanya berlaku untuk provider/profil model ini, supaya kunci satu provider tidak
    /// terkirim ke alamat provider lain saat berganti model.
    #[cfg(test)]
    pub fn new(spec: &str, url: Option<String>, ctx: Option<u64>) -> Result<Provider, String> {
        Self::with_options(spec, url, None, ctx, BTreeMap::new())
    }

    pub fn with_options(
        spec: &str,
        url: Option<String>,
        cli_key: Option<String>,
        ctx: Option<u64>,
        mut session_apis: BTreeMap<String, api::Profile>,
    ) -> Result<Provider, String> {
        let cfg = api::Config::load();
        let (profile, kind, model) = resolve_spec(spec, &cfg, &session_apis)?;
        if model.is_empty() || model.contains(char::is_control) {
            return Err(format!("nama model tidak valid: {spec:?}"));
        }
        let has_url = url.is_some();
        let disk = cfg.providers.get(&profile);
        let pre_url = api::preset(&profile).map(|(_, u)| u);
        if let Some(k) = cli_key.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let entry = session_apis.entry(profile.clone()).or_insert_with(|| api::Profile {
                protocol: kind,
                url: disk.and_then(|p| p.url.clone()).or_else(|| pre_url.clone()),
                key: None,
                model: disk.and_then(|p| p.model.clone()),
            });
            entry.key = Some(k.to_string());
            if let Some(u) = &url {
                entry.url = Some(u.trim_end_matches('/').to_string());
            }
        }
        let sess = session_apis.get(&profile);
        let builtin = api::is_builtin(&profile);

        let base = url
            .or_else(|| sess.and_then(|p| p.url.clone()))
            .or_else(|| {
                if builtin {
                    env_url(kind).or_else(|| disk.and_then(|p| p.url.clone()))
                } else {
                    disk.and_then(|p| p.url.clone()).or(pre_url).or_else(|| env_url(kind))
                }
            })
            .unwrap_or_else(|| api::default_url(kind).into());

        let (api_key, auth_bearer) = if kind == Kind::Ollama {
            (None, false)
        } else if let Some(k) = sess.and_then(|p| p.key.clone()) {
            (Some(k), false)
        } else if builtin {
            let (ek, bearer) = env_key(kind);
            if ek.is_some() {
                (ek, bearer)
            } else {
                (disk.and_then(|p| p.key.clone()), false)
            }
        } else if let Some(k) = disk.and_then(|p| p.key.clone()) {
            (Some(k), false)
        } else if let Some(f) = crate::setup::find(&profile) {
            // Provider gratis: hanya dari env miliknya sendiri, misalnya GROQ_API_KEY. Kunci
            // OPENAI_API_KEY tidak boleh terkirim ke provider lain.
            (f.env.iter().find_map(|e| env(e)), false)
        } else {
            env_key(kind)
        };

        let p = Provider {
            kind,
            profile,
            base: base.trim_end_matches('/').into(),
            model,
            ctx: ctx.unwrap_or(match kind {
                Kind::Ollama => OLLAMA_CTX,
                Kind::Anthropic | Kind::Gemini => CLOUD_CTX,
                Kind::OpenAi => 128_000,
            }),
            max_out: 32_000,
            think: false,
            effort: None,
            tools: Tools::Auto,
            text_tools: false,
            echo: false,
            no_think: false,
            detected: false,
            temperature: None,
            json: false,
            failed: None,
            session_apis,
            api_key,
            auth_bearer,
            has_url,
            ctx_user: ctx,
            probed: false,
        };
        p.headers()?;
        Ok(p)
    }

    /// Model lain dengan setelan yang sama.
    pub fn switch(&self, spec: &str) -> Result<Provider, String> {
        let cfg = api::Config::load();
        let (target_profile, _, _) = resolve_spec(spec, &cfg, &self.session_apis)?;
        let same = target_profile == self.profile && self.has_url;
        let mut p = Provider::with_options(
            spec,
            same.then(|| self.base.clone()),
            None,
            self.ctx_user,
            self.session_apis.clone(),
        )?;
        (p.think, p.effort) = (self.think, self.effort.clone());
        p.set_tools(self.tools);
        Ok(p)
    }

    pub fn set_tools(&mut self, t: Tools) {
        (self.tools, self.text_tools) = (t, t == Tools::Text);
    }

    pub fn label(&self) -> String {
        if self.profile == "ollama" {
            self.model.clone()
        } else {
            format!("{}/{}", self.profile, self.model)
        }
    }

    /// Ringkasan kunci aktif (tersamarkan) untuk tampilan `/api`.
    pub fn masked_key(&self) -> String {
        if self.kind == Kind::Ollama {
            return "-".into();
        }
        match &self.api_key {
            Some(k) if self.key_withheld() => format!("{} (ditahan: http non-localhost)", api::mask(k)),
            Some(k) => api::mask(k),
            None => "(kosong)".into(),
        }
    }

    /// Host tujuan bila data meninggalkan komputer ini.
    pub fn remote(&self) -> Option<&str> {
        (!http::is_local(&self.base)).then_some(&self.base)
    }

    /// Env tempat kunci profil ini dibaca: provider gratis memakai env miliknya (GROQ_API_KEY, ...).
    fn key_name(&self) -> &'static str {
        let free = crate::setup::find(&self.profile).filter(|_| !api::is_builtin(&self.profile));
        free.map_or(env_key_name(self.kind), |f| f.env[0])
    }

    pub fn headers(&self) -> Result<Vec<(&'static str, String)>, String> {
        if self.kind == Kind::Ollama {
            return Ok(vec![]);
        }
        let free = crate::setup::find(&self.profile).filter(|_| !api::is_builtin(&self.profile));
        let name = self.key_name();
        let h = self.api_key.as_ref().map(|k| match self.kind {
            Kind::Anthropic if self.auth_bearer => ("Authorization", format!("Bearer {k}")),
            Kind::Anthropic => ("x-api-key", k.clone()),
            Kind::OpenAi => ("Authorization", format!("Bearer {k}")),
            Kind::Gemini => ("x-goog-api-key", k.clone()),
            Kind::Ollama => unreachable!(),
        });
        let local = http::is_local(&self.base);
        let lan = http::is_lan(&self.base);
        let mut v = vec![];
        match h {
            // Server di jaringan lokal (vLLM, LM Studio): jalan tanpa kunci, kunci tidak dikirim.
            Some(_) if self.key_withheld() => {}
            Some(_) if !local && !http::is_tls(&self.base) => {
                return Err(format!("{name} tidak dikirim lewat http ke {}; pakai https", self.base));
            }
            Some(h) => v.push(h),
            // Provider yang memang bisa dipakai tanpa akun (Kilo).
            None if free.is_some_and(|f| f.keyless) => {}
            None if !local && !lan => {
                return Err(format!(
                    "{name} belum diatur untuk {} (set env atau pakai `llmy api set` / `/api`)",
                    self.label()
                ));
            }
            None => {}
        }
        if self.kind == Kind::Anthropic {
            v.push(("anthropic-version", anthropic::VERSION.into()));
        }
        Ok(v)
    }

    /// http biasa ke jaringan lokal: kunci akan terbaca di jalan, jadi tidak dikirim.
    fn key_withheld(&self) -> bool {
        http::is_lan(&self.base) && !http::is_tls(&self.base)
    }

    pub fn chat(
        &mut self,
        msgs: &[Value],
        tools: &Value,
        mode: Mode,
        cancel: &Cancel,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Reply, String> {
        self.failed = None;
        self.detect(cancel, on);
        if !self.text_tools {
            match self.send(msgs, tools, mode, cancel, on) {
                Err(e) if self.tools == Tools::Auto && detect::refuses(&e) && !cancel.is_set() => self.use_text(on),
                r => return r.map_err(|e| self.explain(e)),
            }
        }
        let mut f = text::Filter::default();
        let r = self.send(&text::convert(msgs, tools, mode), &Value::Null, mode, cancel, &mut |d| match d {
            Delta::Text(s) => f.push(s, on),
            d => on(d),
        });
        let mut r = r.map_err(|e| self.explain(e))?;
        f.finish(&mut r, on);
        Ok(r)
    }

    /// Satu permintaan, tanpa diulang.
    pub fn once(&self, msgs: &[Value], tools: &Value, mode: Mode, cancel: &Cancel, on: &mut dyn FnMut(Delta)) -> Result<Reply, Fail> {
        match self.kind {
            Kind::Ollama => ollama::chat(self, msgs, tools, mode, cancel, on),
            Kind::Anthropic => anthropic::chat(self, msgs, tools, mode, cancel, on),
            Kind::OpenAi => openai::chat(self, msgs, tools, mode, cancel, on),
            Kind::Gemini => gemini::chat(self, msgs, tools, mode, cancel, on),
        }
    }

    fn send(
        &mut self,
        msgs: &[Value],
        tools: &Value,
        mode: Mode,
        cancel: &Cancel,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Reply, Fail> {
        self.probe(cancel);
        let mut owned: Option<Vec<Value>> = None;
        let mut attempt = 0;
        loop {
            let mut emitted = false;
            let m = owned.as_deref().unwrap_or(msgs);
            let mut tap = |d: Delta| {
                emitted = true;
                on(d);
            };
            let e = match self.once(m, tools, mode, cancel, &mut tap) {
                Ok(mut r) => {
                    r.stripped = owned.is_some();
                    return Ok(r);
                }
                Err(e) => e,
            };
            self.failed = Some(e.status);
            if cancel.is_set() || emitted {
                return Err(e);
            }
            // Riwayat berubah sejak thinking dibuat: buang thinking, coba sekali lagi.
            if e.status == 400 && owned.is_none() && e.msg.contains("bound to a different conversation") {
                let mut v = msgs.to_vec();
                anthropic::strip_thinking(&mut v);
                owned = Some(v);
                continue;
            }
            if self.adapt(&e, on) {
                continue;
            }
            let wait = e.retry_after.unwrap_or(1 << attempt);
            if !self.retryable(&e) || attempt >= RETRIES || wait > 60 {
                self.failed = Some(e.status);
                return Err(e);
            }
            attempt += 1;
            on(Delta::Work("Server sibuk, mencoba lagi"));
            let until = Instant::now() + Duration::from_secs(wait);
            while Instant::now() < until {
                if cancel.is_set() {
                    return Err(Fail::from("dibatalkan".to_string()));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    /// Server kompatibel (DeepSeek, Kimi) bisa mewajibkan thinking dikirim balik. Kirim; bila
    /// masih ditolak (mis. bagian riwayat tanpa thinking), matikan thinking.
    fn adapt(&mut self, e: &Fail, on: &mut dyn FnMut(Delta)) -> bool {
        let about = match self.kind {
            Kind::OpenAi => e.msg.contains("reasoning_content"),
            Kind::Anthropic => !self.official() && e.msg.contains("thinking"),
            Kind::Ollama | Kind::Gemini => false,
        };
        if !matches!(e.status, 400 | 422) || !about || self.no_think {
            return false;
        }
        if self.kind == Kind::OpenAi && !self.echo {
            self.echo = true;
            self.learn("reasoning", true.into());
        } else {
            self.no_think = true;
            on(Delta::Note("Server menolak riwayat thinking; thinking dimatikan untuk model ini."));
        }
        true
    }

    /// Kegagalan yang layak dialihkan ke provider cadangan: kuota/batas habis, server
    /// bermasalah, atau server tak terjangkau. Bukan kunci salah atau permintaan salah.
    pub fn exhausted(&self) -> bool {
        matches!(self.failed, Some(402 | 408 | 409 | 429 | 500..=599))
            || self.failed == Some(0) && self.kind != Kind::Ollama
    }

    fn retryable(&self, e: &Fail) -> bool {
        // Ollama gagal mengurai tool call yang ditulis model (HTTP 500): jawabannya acak, jadi
        // diulang bisa berhasil.
        if self.kind == Kind::Ollama {
            return e.status == 500 && (e.msg.contains("syntax error") || e.msg.contains("parsing tool call") || e.msg.contains("parse tool call"));
        }
        // Ollama lokal: gagal terhubung berarti memang mati, tak perlu menunggu.
        self.kind != Kind::Ollama
            && !e.no_retry
            && !e.msg.contains("enforced_spend_limit_reached")
            && !e.msg.contains("insufficient_quota")
            && (matches!(e.status, 408 | 409 | 429 | 500..=599) || e.status == 0 && self.remote().is_some())
    }

    fn explain(&self, e: Fail) -> String {
        let hint: String = match (e.status, self.kind) {
            (401 | 403, _) if self.kind != Kind::Ollama && self.key_withheld() => " (kunci tidak dikirim lewat http; pakai https atau terowongan ssh)".into(),
            _ if !self.text_tools && detect::refuses(&e) => " (coba --tools text)".into(),
            (401 | 403, Kind::Anthropic | Kind::OpenAi | Kind::Gemini) => format!(" (periksa {} atau /api)", self.key_name()),
            (404, _) => " (periksa nama model: /model)".into(),
            _ => String::new(),
        };
        e.msg + &hint
    }

    /// Ambil batas model dari API sekali; gagal bukan masalah, pakai bawaan.
    fn probe(&mut self, cancel: &Cancel) {
        if self.probed {
            return;
        }
        match self.kind {
            Kind::Anthropic => {
                self.probed = true;
                let path = format!("/v1/models/{}", self.model);
                if let Ok(v) = self.get_json(&path, cancel) {
                    if let Some(n) = v["max_tokens"].as_u64().filter(|n| *n >= 1024) {
                        self.max_out = n.min(64_000);
                    }
                    if let (None, Some(n)) = (self.ctx_user, v["max_input_tokens"].as_u64().filter(|n| *n >= 4096)) {
                        self.ctx = n.min(CLOUD_CTX);
                    }
                }
            }
            Kind::Gemini => {
                self.probed = true;
                let m = self.model.strip_prefix("models/").unwrap_or(&self.model);
                let path = format!("/models/{m}");
                if let Ok(v) = self.get_json(&path, cancel) {
                    if let Some(n) = v["outputTokenLimit"].as_u64().filter(|n| *n >= 1024) {
                        self.max_out = n.min(64_000);
                    }
                    if let (None, Some(n)) = (self.ctx_user, v["inputTokenLimit"].as_u64().filter(|n| *n >= 4096)) {
                        self.ctx = n.min(CLOUD_CTX);
                    }
                }
            }
            Kind::Ollama | Kind::OpenAi => {}
        }
    }

    pub fn get_json(&self, path: &str, cancel: &Cancel) -> Result<Value, String> {
        let mut raw = String::new();
        let headers = self.headers()?;
        let req = Req { base: &self.base, method: "GET", path, headers: &headers, body: "" };
        http::request(&req, cancel, &mut |l| {
            raw.push_str(l);
            Ok(())
        })
        .map_err(|e| self.explain(e))?;
        serde_json::from_str(&raw).map_err(|e| e.to_string())
    }

    /// Daftar model dari provider ini. Dukungan tool tidak dilaporkan, jadi semua ditampilkan.
    pub fn models(&self, cancel: &Cancel) -> Result<Vec<String>, String> {
        if self.kind == Kind::Gemini {
            return gemini::models(self, cancel);
        }
        let (path, list, name) = match self.kind {
            Kind::Ollama => ("/api/tags", "models", "name"),
            Kind::Anthropic => ("/v1/models?limit=1000", "data", "id"),
            Kind::OpenAi => ("/models", "data", "id"),
            Kind::Gemini => unreachable!(),
        };
        let v = self.get_json(path, cancel)?;
        let list = v[list].as_array().cloned().unwrap_or_default();
        Ok(list.iter().filter_map(|m| m[name].as_str().map(String::from)).collect())
    }
}

/// Isi satu baris SSE: `data: {...}`. Baris lain (event, id, komentar) diabaikan.
pub fn sse(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(str::trim).filter(|s| !s.is_empty() && *s != "[DONE]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilih_provider() {
        assert_eq!(split("anthropic/claude-opus-5-5"), (Kind::Anthropic, "claude-opus-5-5"));
        assert_eq!(split("claude-opus-5-5"), (Kind::Anthropic, "claude-opus-5-5"));
        assert_eq!(split("gemini/gemini-2.5-flash"), (Kind::Gemini, "gemini-2.5-flash"));
        assert_eq!(split("gemini-2.5-flash"), (Kind::Gemini, "gemini-2.5-flash"));
        assert_eq!(split("openai/a/b"), (Kind::OpenAi, "a/b"));
        assert_eq!(split("qwen3.5:9b"), (Kind::Ollama, "qwen3.5:9b"));
        assert_eq!(split("hf.co/x/y"), (Kind::Ollama, "hf.co/x/y"));
        assert!(Provider::new("openai/", None, None).is_err());
    }

    #[test]
    fn petunjuk_401_menyebut_env_kunci_profil() {
        let f = |spec: &str| {
            let (name, model) = spec.split_once('/').unwrap();
            let mut sess = BTreeMap::new();
            sess.insert(name.into(), api::Profile { protocol: Kind::OpenAi, url: None, key: Some("kunci-uji-123456".into()), model: Some(model.into()) });
            let p = Provider::with_options(name, None, None, None, sess).unwrap();
            p.explain(Fail { status: 401, msg: "HTTP 401: ditolak".into(), retry_after: None, no_retry: false, location: None })
        };
        assert!(f("zai/glm-4.7-flash").ends_with("(periksa ZAI_API_KEY atau /api)"));
        assert!(f("groq/openai/gpt-oss-120b").contains("GROQ_API_KEY"));
        assert!(f("openai/gpt-x").contains("OPENAI_API_KEY"));
    }

    #[test]
    fn url_tidak_terbawa_ke_provider_lain() {
        let p = Provider::new("ollama/x", Some("http://10.0.0.9:11434".into()), None).unwrap();
        assert_eq!(p.switch("y").unwrap().base, "http://10.0.0.9:11434");
        let q = Provider::new("openai/x", Some("http://localhost:1/v1/".into()), None).unwrap();
        assert_eq!(q.base, "http://localhost:1/v1");
        assert!(q.switch("y").unwrap().base != q.base);
        assert!(q.remote().is_none() && p.remote().is_some());
    }

    #[test]
    fn sesi_api_dan_profil_terisolasi() {
        let mut sess = BTreeMap::new();
        sess.insert("deepseek".into(), api::Profile {
            protocol: Kind::OpenAi,
            url: Some("https://api.deepseek.com".into()),
            key: Some("sk-ds-123456789".into()),
            model: Some("deepseek-chat".into()),
        });
        let p = Provider::with_options("deepseek", None, None, None, sess).unwrap();
        assert_eq!(p.kind, Kind::OpenAi);
        assert_eq!(p.profile, "deepseek");
        assert_eq!(p.base, "https://api.deepseek.com");
        assert_eq!(p.model, "deepseek-chat");
        assert_eq!(p.label(), "deepseek/deepseek-chat");
        // Pindah ke provider lain tidak membawa url/kunci deepseek.
        let o = p.switch("ollama/qwen3.5:9b").unwrap();
        assert_eq!(o.kind, Kind::Ollama);
        assert_eq!(o.base, "http://localhost:11434");
        assert!(o.api_key.is_none());
    }

    #[test]
    fn baris_sse() {
        assert_eq!(sse("data: {}"), Some("{}"));
        assert_eq!(sse("data:{}"), Some("{}"));
        for l in ["event: x", ": ping", "data: [DONE]", "data:"] {
            assert_eq!(sse(l), None);
        }
    }
}
