//! Klien Ollama native (/api/chat). Sengaja bukan /v1: template model lama bisa
//! salah mengurutkan pesan tanpa error.

use crate::cancel::Cancel;
use crate::http;
use serde_json::{Value, json};

pub struct Provider {
    pub base: String,
    pub model: String,
    pub ctx: u64,
    pub think: bool,
}

pub enum Delta<'a> {
    Text(&'a str),
    Think,
}

#[derive(Default)]
pub struct Reply {
    pub content: String,
    pub thinking: String,
    pub calls: Vec<Value>,
    pub tokens: u64,
    pub done_reason: String,
}

impl Provider {
    pub fn chat(
        &self,
        msgs: &[Value],
        tools: &Value,
        cancel: &Cancel,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Reply, String> {
        let think = self.think && tools.is_array();
        match self.chat_once(msgs, tools, Some(think), cancel, on) {
            // Model tanpa thinking menolak field `think`; penolakan datang sebelum token apa pun.
            Err(e) if e.contains("does not support thinking") => {
                self.chat_once(msgs, tools, None, cancel, on)
            }
            r => r,
        }
    }

    fn chat_once(
        &self,
        msgs: &[Value],
        tools: &Value,
        think: Option<bool>,
        cancel: &Cancel,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Reply, String> {
        let mut body = json!({
            "model": self.model, "messages": msgs, "stream": true,
            "options": {"num_ctx": self.ctx},
        });
        if tools.is_array() {
            body["tools"] = tools.clone();
        } else {
            // Tanpa tool = meringkas: batasi panjangnya.
            body["options"]["num_predict"] = json!(1200);
        }
        if let Some(t) = think {
            body["think"] = json!(t);
        }
        let mut out = Reply::default();
        let mut done = false;
        http::request(&self.base, "POST", "/api/chat", &body.to_string(), cancel, &mut |l| {
            let v: Value = serde_json::from_str(l).map_err(|e| format!("JSON model: {e}"))?;
            if let Some(e) = v["error"].as_str() {
                return Err(e.to_string());
            }
            let m = &v["message"];
            if let Some(s) = m["thinking"].as_str().filter(|s| !s.is_empty()) {
                out.thinking.push_str(s);
                on(Delta::Think);
            }
            if let Some(s) = m["content"].as_str().filter(|s| !s.is_empty()) {
                out.content.push_str(s);
                on(Delta::Text(s));
            }
            if let Some(c) = m["tool_calls"].as_array() {
                out.calls.extend(c.iter().cloned());
            }
            if v["done"].as_bool() == Some(true) {
                done = true;
                out.done_reason = v["done_reason"].as_str().unwrap_or("").into();
                out.tokens = v["prompt_eval_count"].as_u64().unwrap_or(0)
                    + v["eval_count"].as_u64().unwrap_or(0);
            }
            Ok(())
        })?;
        if !done {
            return Err("koneksi terputus sebelum jawaban selesai".into());
        }
        Ok(out)
    }

    /// Nama model yang mendukung tool calling tidak dilaporkan /api/tags; tampilkan semua.
    pub fn models(&self, cancel: &Cancel) -> Result<Vec<String>, String> {
        let mut raw = String::new();
        http::request(&self.base, "GET", "/api/tags", "", cancel, &mut |l| {
            raw.push_str(l);
            Ok(())
        })?;
        let v: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let list = v["models"].as_array().cloned().unwrap_or_default();
        Ok(list.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect())
    }
}
