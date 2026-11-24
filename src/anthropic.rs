//! Claude Messages API (HTTP langsung, streaming).
//!
//! Blok jawaban asli disimpan di `blocks` dan dikirim balik persis sama: thinking terikat
//! pada riwayat yang menghasilkannya. Setiap kali riwayat lama diubah (dipangkas,
//! diringkas, sesi dilanjutkan), thinking lama harus dibuang lewat `strip_thinking`.

use crate::cancel::Cancel;
use crate::http::{self, Fail, Req};
use crate::provider::{Delta, Mode, Provider, Reply, sse};
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};

pub const BASE: &str = "https://api.anthropic.com";
pub const VERSION: &str = "2026-10-21";
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

/// Model yang bisa menolak lewat pengklasifikasi dan punya fallback di sisi server.
fn has_fallback(model: &str) -> bool {
    ["claude-opus-5", "claude-sonnet-5-5", "claude-fable-5"].iter().any(|p| model.starts_with(p))
}

fn is_thinking(b: &Value) -> bool {
    matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking"))
}

pub fn strip_thinking(msgs: &mut [Value]) {
    for m in msgs {
        if let Some(b) = m.get_mut("blocks").and_then(Value::as_array_mut) {
            b.retain(|b| !is_thinking(b));
        }
    }
}

fn valid_id(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Riwayat netral -> (system, messages). Pesan berurutan dengan role sama digabung,
/// hasil tool dipasangkan menurut urutan.
fn convert(msgs: &[Value]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut out: Vec<(&str, Vec<Value>)> = vec![];
    let mut wait: VecDeque<String> = VecDeque::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (i, m) in msgs.iter().enumerate() {
        let text = m["content"].as_str().unwrap_or("");
        let (role, blocks) = match m["role"].as_str() {
            Some("system") => {
                system.push_str(text);
                continue;
            }
            Some("assistant") => {
                let mut v: Vec<Value> = vec![];
                if let Some(b) = m["blocks"].as_array() {
                    v = b.clone();
                    for c in v.iter().filter(|c| c["type"] == "tool_use") {
                        let id = c["id"].as_str().unwrap_or("").to_string();
                        seen.insert(id.clone());
                        wait.push_back(id);
                    }
                } else {
                    if !text.trim().is_empty() {
                        v.push(json!({"type":"text","text":text}));
                    }
                    for (j, c) in m["tool_calls"].as_array().into_iter().flatten().enumerate() {
                        let mut id = c["id"].as_str().filter(|s| valid_id(s)).unwrap_or("").to_string();
                        if id.is_empty() || !seen.insert(id.clone()) {
                            id = format!("call_{i}_{j}");
                            seen.insert(id.clone());
                        }
                        let f = &c["function"];
                        let input = if f["arguments"].is_object() { f["arguments"].clone() } else { json!({}) };
                        v.push(json!({"type":"tool_use","id":id,"name":f["name"],"input":input}));
                        wait.push_back(id);
                    }
                }
                ("assistant", v)
            }
            Some("tool") => {
                let Some(id) = wait.pop_front() else { continue };
                let mut b = json!({"type":"tool_result","tool_use_id":id,"content":text});
                if text.starts_with("Error: ") {
                    b["is_error"] = json!(true);
                }
                ("user", vec![b])
            }
            _ if text.trim().is_empty() => continue,
            _ => ("user", vec![json!({"type":"text","text":text})]),
        };
        if blocks.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some((r, b)) if *r == role => b.extend(blocks),
            _ => out.push((role, blocks)),
        }
    }
    // Titik cache di akhir percakapan; yang di system dipasang pemanggil.
    if let Some(b) = out.last_mut().and_then(|(_, b)| b.last_mut()).filter(|b| !is_thinking(b)) {
        b["cache_control"] = json!({"type":"ephemeral"});
    }
    (system, out.into_iter().map(|(r, b)| json!({"role":r,"content":b})).collect())
}

fn convert_tools(tools: &Value, eager: bool) -> Vec<Value> {
    tools.as_array().into_iter().flatten().map(|t| {
        let f = &t["function"];
        let mut o = json!({"name":f["name"],"description":f["description"],"input_schema":f["parameters"]});
        if eager {
            o["eager_input_streaming"] = json!(true);
        }
        o
    }).collect()
}

fn body(p: &Provider, msgs: &[Value], tools: &Value, mode: Mode) -> Value {
    let official = p.base == BASE;
    let (system, messages) = convert(msgs);
    let mut b = json!({
        "model": p.model,
        "max_tokens": if mode == Mode::Turn { p.max_out } else { p.max_out.min(8_000) },
        "stream": true,
        "system": [{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}],
        // Sama di setiap permintaan, juga saat meringkas: daftar tool bagian dari riwayat.
        "tools": convert_tools(tools, official),
        "messages": messages,
    });
    if mode == Mode::Summary {
        b["tool_choice"] = json!({"type":"none"});
    }
    if let Some(e) = &p.effort {
        b["output_config"] = json!({"effort": e});
    }
    if p.think {
        b["thinking"] = json!({"type":"adaptive","display":"summarized"});
    }
    if official && has_fallback(&p.model) {
        b["fallbacks"] = json!("default");
    }
    b
}

/// Menyusun blok dari peristiwa SSE.
#[derive(Default)]
struct Stream {
    blocks: Vec<Value>,
    partial: Vec<String>,
    /// Argumen tool yang bukan JSON sah, menurut indeks blok.
    bad: Vec<(usize, String)>,
    usage: [u64; 4],
    stop: String,
    details: String,
    done: bool,
    /// Kesalahan server di tengah stream yang layak diulang.
    overloaded: bool,
}

const USAGE: [&str; 4] = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"];

impl Stream {
    fn usage(&mut self, u: &Value) {
        for (i, k) in USAGE.iter().enumerate() {
            if let Some(n) = u[k].as_u64() {
                self.usage[i] = n;
            }
        }
    }

    fn event(&mut self, v: &Value, on: &mut dyn FnMut(Delta)) -> Result<(), String> {
        let i = v["index"].as_u64().unwrap_or(0) as usize;
        match v["type"].as_str().unwrap_or("") {
            "message_start" => self.usage(&v["message"]["usage"]),
            "content_block_start" => {
                if i != self.blocks.len() || i > 512 {
                    return Err("urutan blok tidak valid".into());
                }
                self.blocks.push(v["content_block"].clone());
                self.partial.push(String::new());
            }
            "content_block_delta" => {
                let Some(b) = self.blocks.get_mut(i) else { return Err("delta tanpa blok".into()) };
                let d = &v["delta"];
                let add = |b: &mut Value, k: &str| {
                    let s = format!("{}{}", b[k].as_str().unwrap_or(""), d[k].as_str().unwrap_or(""));
                    b[k] = json!(s);
                };
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        add(b, "text");
                        on(Delta::Text(d["text"].as_str().unwrap_or("")));
                    }
                    "thinking_delta" => {
                        add(b, "thinking");
                        on(Delta::Think);
                    }
                    "signature_delta" => add(b, "signature"),
                    "input_json_delta" => self.partial[i].push_str(d["partial_json"].as_str().unwrap_or("")),
                    _ => {}
                }
            }
            "content_block_stop" => {
                let Some(b) = self.blocks.get_mut(i) else { return Ok(()) };
                let raw = &self.partial[i];
                if b["type"] == "tool_use" && !raw.trim().is_empty() {
                    match serde_json::from_str::<Value>(raw) {
                        Ok(v) if v.is_object() => b["input"] = v,
                        _ => {
                            b["input"] = json!({"INVALID_JSON": raw});
                            self.bad.push((i, raw.clone()));
                        }
                    }
                }
            }
            "message_delta" => {
                self.usage(&v["usage"]);
                if let Some(s) = v["delta"]["stop_reason"].as_str() {
                    self.stop = s.into();
                }
                let d = &v["delta"]["stop_details"];
                let part = |k: &str| d[k].as_str().unwrap_or("").to_string();
                if d.is_object() {
                    self.details = format!("{} {}", part("category"), part("explanation")).trim().into();
                }
            }
            "message_stop" => self.done = true,
            "error" => {
                let k = v["error"]["type"].as_str().unwrap_or("");
                self.overloaded = matches!(k, "overloaded_error" | "api_error" | "rate_limit_error");
                return Err(v["error"]["message"].as_str().unwrap_or("kesalahan server").to_string());
            }
            _ => {}
        }
        Ok(())
    }

    fn reply(mut self) -> Reply {
        let mut out = Reply { tokens: self.usage.iter().sum(), ..Default::default() };
        // Setelah fallback di tengah jawaban: thinking dan tool_use sebelum batas tidak
        // boleh dikirim balik, dan tool-nya tidak boleh dijalankan.
        if let Some(f) = self.blocks.iter().rposition(|b| b["type"] == "fallback") {
            out.note = self.blocks[f]["to"]["model"].as_str().map(|m| format!("Dijawab oleh model cadangan {m}."));
            let mut keep: Vec<bool> = vec![];
            for (i, b) in self.blocks.iter().enumerate() {
                keep.push(i >= f || matches!(b["type"].as_str(), Some("text" | "fallback")));
            }
            let mut n = 0;
            for (i, k) in keep.iter().enumerate() {
                if !k {
                    self.bad.retain(|b| b.0 != i);
                }
            }
            self.blocks.retain(|_| {
                n += 1;
                keep[n - 1]
            });
        }
        // Blok teks kosong ditolak bila dikirim balik.
        self.blocks.retain(|b| b["type"] != "text" || b["text"].as_str().is_some_and(|s| !s.is_empty()));
        for b in &self.blocks {
            match b["type"].as_str() {
                Some("text") => out.content.push_str(b["text"].as_str().unwrap_or("")),
                Some("tool_use") => {
                    // Argumen rusak diteruskan sebagai teks supaya ditolak sebelum dijalankan.
                    let a = match b["input"]["INVALID_JSON"].as_str() {
                        Some(raw) if self.bad.iter().any(|(_, r)| r == raw) => json!(raw),
                        _ => b["input"].clone(),
                    };
                    out.calls.push(json!({"id":b["id"],"function":{"name":b["name"],"arguments":a}}));
                }
                _ => {}
            }
        }
        out.done_reason = match self.stop.as_str() {
            "max_tokens" | "model_context_window_exceeded" => "length".into(),
            s => s.to_string(),
        };
        if self.stop == "refusal" {
            out.note = Some(self.details);
        }
        out.blocks = Some(json!(self.blocks));
        out
    }
}

pub fn chat(p: &Provider, msgs: &[Value], tools: &Value, mode: Mode, cancel: &Cancel, on: &mut dyn FnMut(Delta)) -> Result<Reply, Fail> {
    let body = body(p, msgs, tools, mode);
    let mut headers = p.headers()?;
    if !body["fallbacks"].is_null() {
        headers.push(("anthropic-beta", FALLBACK_BETA.into()));
    }
    let mut s = Stream::default();
    let req = Req { base: &p.base, method: "POST", path: "/v1/messages", headers: &headers, body: &body.to_string() };
    let r = http::request(&req, cancel, &mut |l| match sse(l) {
        Some(d) => s.event(&serde_json::from_str(d).map_err(|e| format!("JSON model: {e}"))?, on),
        None => Ok(()),
    });
    if let Err(mut e) = r {
        if s.overloaded {
            e.status = 529;
        }
        return Err(e);
    }
    if !s.done {
        return Err("koneksi terputus sebelum jawaban selesai".to_string().into());
    }
    Ok(s.reply())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(ev: &[Value]) -> (Result<(), String>, Stream, String) {
        let (mut s, mut t) = (Stream::default(), String::new());
        for v in ev {
            let r = s.event(v, &mut |d| if let Delta::Text(x) = d { t.push_str(x) });
            if r.is_err() {
                return (r, s, t);
            }
        }
        (Ok(()), s, t)
    }
    fn start(i: u64, b: Value) -> Value {
        json!({"type":"content_block_start","index":i,"content_block":b})
    }
    fn delta(i: u64, d: Value) -> Value {
        json!({"type":"content_block_delta","index":i,"delta":d})
    }
    fn stop(i: u64) -> Value {
        json!({"type":"content_block_stop","index":i})
    }

    #[test]
    fn stream_lengkap() {
        let (r, s, t) = run(&[
            json!({"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":1}}}),
            json!({"type":"ping"}),
            start(0, json!({"type":"thinking","thinking":"","signature":""})),
            delta(0, json!({"type":"thinking_delta","thinking":"hm"})),
            delta(0, json!({"type":"signature_delta","signature":"SIG"})),
            stop(0),
            start(1, json!({"type":"text","text":""})),
            delta(1, json!({"type":"text_delta","text":"ha"})),
            delta(1, json!({"type":"text_delta","text":"lo"})),
            stop(1),
            start(2, json!({"type":"tool_use","id":"t1","name":"read","input":{}})),
            delta(2, json!({"type":"input_json_delta","partial_json":"{\"pa"})),
            delta(2, json!({"type":"input_json_delta","partial_json":"th\":\"a\"}"})),
            stop(2),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}),
            json!({"type":"message_stop"}),
        ]);
        assert!(r.is_ok() && s.done);
        let o = s.reply();
        assert_eq!((t.as_str(), o.content.as_str(), o.tokens), ("halo", "halo", 120));
        assert_eq!(o.calls[0], json!({"id":"t1","function":{"name":"read","arguments":{"path":"a"}}}));
        assert_eq!(o.blocks.unwrap()[0], json!({"type":"thinking","thinking":"hm","signature":"SIG"}));
    }

    #[test]
    fn argumen_rusak_tidak_jadi_objek() {
        let (_, s, _) = run(&[
            start(0, json!({"type":"tool_use","id":"t","name":"bash","input":{}})),
            delta(0, json!({"type":"input_json_delta","partial_json":"{\"command\":\"ls"})),
            stop(0),
        ]);
        let o = s.reply();
        assert!(o.calls[0]["function"]["arguments"].is_string());
        assert!(o.blocks.unwrap()[0]["input"].is_object());
    }

    #[test]
    fn fallback_membuang_sebelum_batas() {
        let (_, s, _) = run(&[
            start(0, json!({"type":"thinking","thinking":"x","signature":"s"})),
            start(1, json!({"type":"text","text":"awal "})),
            start(2, json!({"type":"tool_use","id":"lama","name":"bash","input":{}})),
            start(3, json!({"type":"fallback","from":{"model":"a"},"to":{"model":"b"}})),
            start(4, json!({"type":"tool_use","id":"baru","name":"read","input":{}})),
        ]);
        let o = s.reply();
        assert_eq!(o.calls.len(), 1);
        assert_eq!(o.calls[0]["id"], "baru");
        let t: Vec<String> = o.blocks.unwrap().as_array().unwrap().iter().map(|b| b["type"].as_str().unwrap().into()).collect();
        assert_eq!(t, ["text", "fallback", "tool_use"]);
        assert!(o.note.unwrap().contains('b'));
    }

    #[test]
    fn konversi_riwayat() {
        let m = [
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"u"}),
            json!({"role":"assistant","content":"x","blocks":[
                {"type":"thinking","thinking":"","signature":"S"},{"type":"tool_use","id":"t1","name":"read","input":{}}],
                "tool_calls":[{"id":"t1","function":{"name":"read","arguments":{}}}]}),
            json!({"role":"tool","content":"Error: gagal"}),
            json!({"role":"user","content":"lagi"}),
            // dari Ollama: tanpa blocks, id berulang
            json!({"role":"assistant","content":"","tool_calls":[{"id":"t1","function":{"name":"bash","arguments":{"a":1}}},{"function":{"name":"bash","arguments":"rusak"}}]}),
            json!({"role":"tool","content":"1"}),
            json!({"role":"tool","content":"2"}),
            json!({"role":"assistant","content":" "}),
        ];
        let (s, o) = convert(&m);
        assert_eq!(s, "sys");
        let roles: Vec<_> = o.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
        assert_eq!(o[1]["content"][0]["signature"], "S");
        assert_eq!(o[2]["content"][0]["is_error"], true);
        assert_eq!(o[2]["content"][1]["text"], "lagi");
        let ids = [&o[3]["content"][0]["id"], &o[3]["content"][1]["id"]];
        assert!(ids[0] != "t1" && ids[0] != ids[1]);
        assert_eq!(&o[4]["content"][1]["tool_use_id"], ids[1]);
        assert_eq!(o[4]["content"][1]["cache_control"]["type"], "ephemeral");
        assert!(o[3]["content"][1]["input"].is_object());
    }

    #[test]
    fn permintaan_ke_server_resmi() {
        // Dibuat lewat localhost agar tak butuh kunci, lalu diarahkan ke alamat resmi.
        let mut p = Provider::new("anthropic/claude-opus-5-5", Some("http://localhost:1".into()), None).unwrap();
        let t = crate::tools::schema();
        let m = [json!({"role":"system","content":"s"}), json!({"role":"user","content":"u"})];
        let lokal = body(&p, &m, &t, Mode::Turn);
        assert!(lokal.get("fallbacks").is_none() && lokal["tools"][0].get("eager_input_streaming").is_none());
        p.base = BASE.into();
        let b = body(&p, &m, &t, Mode::Turn);
        assert_eq!(b["fallbacks"], "default");
        assert!(b["tools"].as_array().unwrap().iter().all(|t| t["eager_input_streaming"] == true && t["input_schema"]["type"] == "object"));
        assert!(b.get("thinking").is_none() && b.get("tool_choice").is_none() && b.get("temperature").is_none());
        assert_eq!(body(&p, &m, &t, Mode::Summary)["tools"], b["tools"]);
        p.model = "claude-haiku-5-5".into();
        assert!(body(&p, &m, &t, Mode::Turn).get("fallbacks").is_none());
    }

    #[test]
    fn strip_hanya_thinking() {
        let mut m = [json!({"role":"assistant","content":"x","blocks":[{"type":"thinking"},{"type":"redacted_thinking"},{"type":"text","text":"x"}]}), json!({"role":"user","content":"u"})];
        strip_thinking(&mut m);
        assert_eq!(m[0]["blocks"], json!([{"type":"text","text":"x"}]));
        assert!(m[1].get("blocks").is_none());
    }
}
