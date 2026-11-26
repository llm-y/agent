//! Mengenali kemampuan server yang tidak dikenal: apakah tool native jalan, dan apakah
//! `reasoning_content` wajib dikirim balik. Hasilnya disimpan per alamat dan model.

use crate::cancel::Cancel;
use crate::http::Fail;
use crate::provider::{Delta, Kind, Mode, Provider, Reply};
use crate::{anthropic, gemini, openai, sys, text};
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tools {
    /// Native bila jalan, teks bila hanya teks yang terbukti jalan.
    Auto,
    Native,
    Text,
}

impl Tools {
    pub fn parse(s: &str) -> Option<Tools> {
        Some(match s {
            "auto" => Tools::Auto,
            "native" => Tools::Native,
            "text" => Tools::Text,
            _ => return None,
        })
    }
}

/// Server dan model bisa berubah, jadi temuan lama diperiksa ulang.
const DAY: u64 = 24 * 3600;
const MAX_DAYS: u64 = 30;
const MAX_ENTRIES: usize = 200;

/// Server menolak field `tools`, bukan menolak isi percakapan.
pub fn refuses(e: &Fail) -> bool {
    let m = e.msg.to_ascii_lowercase();
    matches!(e.status, 400 | 404 | 422 | 500 | 501)
        && ["support tool", "tool choice requires", "requires --jinja", "support function calling"].iter().any(|s| m.contains(s))
}

/// Permintaan periksa yang sama gagal dengan `tools`; jalan tanpa `tools` = `tools` sebabnya.
fn suspect(e: &Fail) -> bool {
    matches!(e.status, 400 | 404 | 422 | 501) || refuses(e)
}

/// Nama yang tidak bisa ditebak dan tidak disebut di pertanyaan: model yang tidak menerima
/// `tools` pun suka menulis panggilan ke nama yang ia baca di pertanyaan.
const NAME: &str = "open_box_q7";

/// Pemeriksaan tidak memuat percakapan pengguna.
fn ask() -> (Vec<Value>, Value) {
    let msgs = vec![
        json!({"role":"system","content":"You are an assistant with tools."}),
        json!({"role":"user","content":"What is in the box labelled \"A\"? Use your tool to look; do not guess."}),
    ];
    let tools = json!([{"type":"function","function":{"name":NAME,"description":"Returns the content of a box.",
        "parameters":{"type":"object","properties":{"label":{"type":"string","description":"Label of the box"}},"required":["label"]}}}]);
    (msgs, tools)
}

fn called(r: &Reply) -> bool {
    r.calls.iter().any(|c| c["function"]["name"] == NAME)
}

fn native(p: &Provider, cancel: &Cancel) -> Result<bool, Fail> {
    let (m, t) = ask();
    Ok(called(&p.once(&m, &t, Mode::Turn, cancel, &mut |_| {})?))
}

fn by_text(p: &Provider, cancel: &Cancel) -> Result<bool, Fail> {
    let (m, t) = ask();
    let mut f = text::Filter::default();
    let mut r = p.once(&text::convert(&m, &t, Mode::Turn), &Value::Null, Mode::Turn, cancel, &mut |d| {
        if let Delta::Text(s) = d {
            f.push(s, &mut |_| {});
        }
    })?;
    f.finish(&mut r, &mut |_| {});
    Ok(called(&r))
}

/// Teks dipilih hanya bila native dua kali gagal dan teks berhasil. `None` = pemeriksaan gagal,
/// "unclear" = model tidak memakai tool dengan cara mana pun.
fn decide(p: &Provider, cancel: &Cancel) -> Option<&'static str> {
    for _ in 0..2 {
        match native(p, cancel) {
            Ok(true) => return Some("native"),
            Ok(false) => {}
            Err(e) if suspect(&e) => break,
            Err(_) => return None,
        }
    }
    for _ in 0..2 {
        if by_text(p, cancel).ok()? {
            return Some("text");
        }
    }
    Some("unclear")
}

impl Provider {
    /// Dipanggil sebelum permintaan pertama.
    pub fn detect(&mut self, cancel: &Cancel, on: &mut dyn FnMut(Delta)) {
        if std::mem::replace(&mut self.detected, true) {
            return;
        }
        let known = load(&self.id());
        self.echo = known["reasoning"] == true;
        if self.tools != Tools::Auto {
            return;
        }
        match known["tools"].as_str() {
            Some(t) => self.text_tools = t == "text",
            // Ollama menolak dengan jelas, dan API resmi pasti mendukung.
            None if self.kind == Kind::Ollama || self.official() => {}
            None => {
                on(Delta::Work("Memeriksa dukungan tool"));
                let d = decide(self, cancel);
                if !cancel.is_set() {
                    match d {
                        Some("text") => self.use_text(on),
                        Some(t) => {
                            self.learn("tools", json!(t));
                            if t == "unclear" {
                                on(Delta::Note("Model ini tidak memakai tool, native maupun teks; membaca dan mengubah file mungkin tidak jalan. Coba model lain."));
                            }
                        }
                        None => {}
                    }
                }
            }
        }
    }

    pub fn use_text(&mut self, on: &mut dyn FnMut(Delta)) {
        self.text_tools = true;
        self.learn("tools", json!("text"));
        on(Delta::Note("Tool calling native tidak jalan di server ini; memakai mode teks (paksa native: --tools native)."));
    }

    pub fn learn(&self, k: &str, v: Value) {
        // Gagal menyimpan hanya berarti diperiksa lagi nanti.
        let _ = save(&self.id(), k, v);
    }

    pub fn official(&self) -> bool {
        match self.kind {
            Kind::Anthropic => self.base == anthropic::BASE,
            Kind::OpenAi => self.base == openai::BASE,
            Kind::Gemini => self.base == gemini::BASE,
            Kind::Ollama => false,
        }
    }

    /// Hash, supaya alamat server tidak tersimpan apa adanya.
    fn id(&self) -> String {
        format!("{:016x}", sys::fnv(format!("{:?}\n{}\n{}", self.kind, self.base, self.model).as_bytes()))
    }
}

fn file() -> Option<PathBuf> {
    sys::state_dir().map(|d| d.join("servers.json"))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn all() -> Value {
    let v = file().and_then(|f| std::fs::read(f).ok()).and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    v.filter(Value::is_object).unwrap_or(json!({}))
}

fn load(id: &str) -> Value {
    let v = all()[id].take();
    let at = v["at"].as_u64().unwrap_or(0);
    let days = if v["tools"] == "unclear" { 1 } else { MAX_DAYS };
    if now().saturating_sub(at) < days * DAY { v } else { Value::Null }
}

fn save(id: &str, k: &str, v: Value) -> std::io::Result<()> {
    let Some(path) = file() else { return Ok(()) };
    let mut m = all();
    let mut e = load(id);
    if !e.is_object() {
        e = json!({});
    }
    e[k] = v;
    e["at"] = json!(now());
    if let Some(o) = m.as_object_mut() {
        o.shift_remove(id);
        // Yang terlama ada di depan.
        while let Some(k) = o.keys().next().filter(|_| o.len() >= MAX_ENTRIES).cloned() {
            o.shift_remove(&k);
        }
        o.insert(id.into(), e);
    }
    // Tulis lalu ganti, supaya proses lain tidak membaca file separuh.
    if let Some(d) = path.parent() {
        sys::private_dir(d)?;
    }
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    sys::private_file(&tmp)?.write_all(m.to_string().as_bytes())?;
    std::fs::rename(&tmp, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mengenali_penolakan_tools() {
        let f = |status, msg: &str| Fail { status, msg: msg.into(), retry_after: None, no_retry: false, location: None };
        for (s, m) in [
            (400, "registry.ollama.ai/library/x:1b does not support tools"),
            (404, "No endpoints found that support tool use"),
            (400, "\"auto\" tool choice requires --enable-auto-tool-choice"),
            (500, "tools param requires --jinja flag"),
        ] {
            assert!(refuses(&f(s, m)), "{m}");
        }
        for (s, m) in [
            (400, "messages with role 'tool' must be a response to a preceding message with 'tool_calls'"),
            (401, "does not support tools"),
            (429, "does not support tools"),
            (0, "does not support tools"),
            (400, "context length exceeded"),
        ] {
            assert!(!refuses(&f(s, m)), "{m}");
        }
        assert!(suspect(&f(422, "extra fields")) && !suspect(&f(500, "overloaded")) && !suspect(&f(401, "x")));
    }
}
