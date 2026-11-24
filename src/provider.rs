//! Memilih dan memanggil model. Riwayat disimpan dalam satu bentuk netral (ala Ollama:
//! role system/user/assistant/tool); tiap provider menerjemahkannya sendiri.

use crate::anthropic;
use crate::cancel::Cancel;
use crate::http::{self, Fail, Req};
use crate::ollama;
use crate::openai;
use crate::text;
use serde_json::Value;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Ollama,
    Anthropic,
    OpenAi,
}

pub enum Delta<'a> {
    Text(&'a str),
    Think,
    /// Menunggu sebelum mencoba lagi.
    Wait,
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
    pub calls: Vec<Value>,
    /// Blok asli Anthropic, harus dikirim balik persis sama.
    pub blocks: Option<Value>,
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
    pub base: String,
    pub model: String,
    pub ctx: u64,
    pub max_out: u64,
    pub think: bool,
    pub effort: Option<String>,
    /// Tool lewat teks, untuk server yang tidak meneruskan `tools`.
    pub text_tools: bool,
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

/// "anthropic/x", "openai/x", "ollama/x"; tanpa awalan: claude-* = Anthropic, lainnya Ollama.
fn split(spec: &str) -> (Kind, &str) {
    for (p, k) in [("anthropic/", Kind::Anthropic), ("openai/", Kind::OpenAi), ("ollama/", Kind::Ollama)] {
        if let Some(m) = spec.strip_prefix(p) {
            return (k, m);
        }
    }
    (if spec.starts_with("claude-") { Kind::Anthropic } else { Kind::Ollama }, spec)
}

impl Provider {
    /// `url` hanya berlaku untuk provider model ini, supaya kunci satu provider tidak
    /// terkirim ke alamat provider lain saat berganti model.
    pub fn new(spec: &str, url: Option<String>, ctx: Option<u64>) -> Result<Provider, String> {
        let (kind, model) = split(spec);
        if model.is_empty() || model.contains(char::is_control) {
            return Err(format!("nama model tidak valid: {spec:?}"));
        }
        let base = url.unwrap_or_else(|| match kind {
            Kind::Ollama => env("LLMY_URL").unwrap_or("http://localhost:11434".into()),
            Kind::Anthropic => env("ANTHROPIC_BASE_URL").unwrap_or(anthropic::BASE.into()),
            Kind::OpenAi => env("OPENAI_BASE_URL").unwrap_or("https://api.openai.com/v1".into()),
        });
        let p = Provider {
            kind,
            base: base.trim_end_matches('/').into(),
            model: model.into(),
            ctx: ctx.unwrap_or(match kind {
                Kind::Ollama => OLLAMA_CTX,
                Kind::Anthropic => CLOUD_CTX,
                Kind::OpenAi => 128_000,
            }),
            max_out: 32_000,
            think: false,
            effort: None,
            text_tools: false,
            ctx_user: ctx,
            probed: false,
        };
        p.headers()?;
        Ok(p)
    }

    /// Model lain dengan setelan yang sama.
    pub fn switch(&self, spec: &str) -> Result<Provider, String> {
        let same = split(spec).0 == self.kind;
        let mut p = Provider::new(spec, same.then(|| self.base.clone()), self.ctx_user)?;
        (p.think, p.effort, p.text_tools) = (self.think, self.effort.clone(), self.text_tools);
        Ok(p)
    }

    pub fn label(&self) -> String {
        match self.kind {
            Kind::Ollama => self.model.clone(),
            Kind::Anthropic => format!("anthropic/{}", self.model),
            Kind::OpenAi => format!("openai/{}", self.model),
        }
    }

    /// Host tujuan bila data meninggalkan komputer ini.
    pub fn remote(&self) -> Option<&str> {
        (!http::is_local(&self.base)).then_some(&self.base)
    }

    pub fn headers(&self) -> Result<Vec<(&'static str, String)>, String> {
        let (h, name) = match self.kind {
            Kind::Ollama => return Ok(vec![]),
            Kind::Anthropic => {
                let h = match (env("ANTHROPIC_API_KEY"), env("ANTHROPIC_AUTH_TOKEN")) {
                    (Some(k), _) => Some(("x-api-key", k)),
                    (None, Some(t)) => Some(("Authorization", format!("Bearer {t}"))),
                    _ => None,
                };
                (h, "ANTHROPIC_API_KEY")
            }
            Kind::OpenAi => (env("OPENAI_API_KEY").map(|k| ("Authorization", format!("Bearer {k}"))), "OPENAI_API_KEY"),
        };
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
            None if !local && !lan => return Err(format!("{name} belum diatur untuk {}", self.label())),
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
        if !self.text_tools {
            return self.send(msgs, tools, mode, cancel, on);
        }
        let mut f = text::Filter::default();
        let mut r = self.send(&text::convert(msgs, tools, mode), &Value::Null, mode, cancel, &mut |d| match d {
            Delta::Text(s) => f.push(s, on),
            d => on(d),
        })?;
        f.finish(&mut r, on);
        Ok(r)
    }

    fn send(
        &mut self,
        msgs: &[Value],
        tools: &Value,
        mode: Mode,
        cancel: &Cancel,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Reply, String> {
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
            let r = match self.kind {
                Kind::Ollama => ollama::chat(self, m, tools, mode, cancel, &mut tap),
                Kind::Anthropic => anthropic::chat(self, m, tools, mode, cancel, &mut tap),
                Kind::OpenAi => openai::chat(self, m, tools, mode, cancel, &mut tap),
            };
            let e = match r {
                Ok(mut r) => {
                    r.stripped = owned.is_some();
                    return Ok(r);
                }
                Err(e) => e,
            };
            if cancel.is_set() || emitted {
                return Err(e.msg);
            }
            // Riwayat berubah sejak thinking dibuat: buang thinking, coba sekali lagi.
            if e.status == 400 && owned.is_none() && e.msg.contains("bound to a different conversation") {
                let mut v = msgs.to_vec();
                anthropic::strip_thinking(&mut v);
                owned = Some(v);
                continue;
            }
            let wait = e.retry_after.unwrap_or(1 << attempt);
            if !self.retryable(&e) || attempt >= RETRIES || wait > 60 {
                return Err(self.explain(e));
            }
            attempt += 1;
            on(Delta::Wait);
            let until = Instant::now() + Duration::from_secs(wait);
            while Instant::now() < until {
                if cancel.is_set() {
                    return Err("dibatalkan".into());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    fn retryable(&self, e: &Fail) -> bool {
        // Ollama lokal: gagal terhubung berarti memang mati, tak perlu menunggu.
        self.kind != Kind::Ollama
            && !e.no_retry
            && !e.msg.contains("enforced_spend_limit_reached")
            && !e.msg.contains("insufficient_quota")
            && (matches!(e.status, 408 | 409 | 429 | 500..=599) || e.status == 0 && self.remote().is_some())
    }

    fn explain(&self, e: Fail) -> String {
        let hint = match (e.status, self.kind) {
            (401 | 403, _) if self.kind != Kind::Ollama && self.key_withheld() => " (kunci tidak dikirim lewat http; pakai https atau terowongan ssh)",
            (400, _) if !self.text_tools && ["support tools", "tool choice requires"].iter().any(|s| e.msg.contains(s)) => {
                " (coba --tools text)"
            }
            (401 | 403, Kind::Anthropic) => " (periksa ANTHROPIC_API_KEY)",
            (401 | 403, Kind::OpenAi) => " (periksa OPENAI_API_KEY)",
            (404, _) => " (periksa nama model: /model)",
            _ => "",
        };
        e.msg + hint
    }

    /// Ambil batas model dari API sekali; gagal bukan masalah, pakai bawaan.
    fn probe(&mut self, cancel: &Cancel) {
        if self.probed || self.kind != Kind::Anthropic {
            return;
        }
        self.probed = true;
        let path = format!("/v1/models/{}", self.model);
        if let Ok(v) = self.get(&path, cancel) {
            if let Some(n) = v["max_tokens"].as_u64().filter(|n| *n >= 1024) {
                self.max_out = n.min(64_000);
            }
            if let (None, Some(n)) = (self.ctx_user, v["max_input_tokens"].as_u64().filter(|n| *n >= 4096)) {
                self.ctx = n.min(CLOUD_CTX);
            }
        }
    }

    fn get(&self, path: &str, cancel: &Cancel) -> Result<Value, String> {
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
        let (path, list, name) = match self.kind {
            Kind::Ollama => ("/api/tags", "models", "name"),
            Kind::Anthropic => ("/v1/models?limit=1000", "data", "id"),
            Kind::OpenAi => ("/models", "data", "id"),
        };
        let v = self.get(path, cancel)?;
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
        assert_eq!(split("openai/a/b"), (Kind::OpenAi, "a/b"));
        assert_eq!(split("qwen3.5:9b"), (Kind::Ollama, "qwen3.5:9b"));
        assert_eq!(split("hf.co/x/y"), (Kind::Ollama, "hf.co/x/y"));
        assert!(Provider::new("openai/", None, None).is_err());
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
    fn baris_sse() {
        assert_eq!(sse("data: {}"), Some("{}"));
        assert_eq!(sse("data:{}"), Some("{}"));
        for l in ["event: x", ": ping", "data: [DONE]", "data:"] {
            assert_eq!(sse(l), None);
        }
    }
}
