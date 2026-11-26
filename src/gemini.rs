//! Google Gemini GenerateContent API (v1beta, HTTP langsung, streaming SSE).

use crate::cancel::Cancel;
use crate::http::{self, Fail, Req};
use crate::provider::{Delta, Mode, Provider, Reply, sse};
use serde_json::{Value, json};
use std::collections::VecDeque;

pub const BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Bersihkan thinking dari bagian `parts` Gemini saat riwayat dipangkas/diringkas.
pub fn strip_thinking(msgs: &mut [Value]) {
    for m in msgs {
        if let Some(p) = m.get_mut("parts").and_then(Value::as_array_mut) {
            p.retain(|x| x["thought"].as_bool() != Some(true));
        }
    }
}

/// Riwayat netral -> (systemInstruction text, contents).
fn convert(msgs: &[Value]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut out: Vec<(&str, Vec<Value>)> = vec![];
    let mut wait: VecDeque<String> = VecDeque::new();
    for m in msgs {
        let text = m["content"].as_str().unwrap_or("");
        let (role, parts) = match m["role"].as_str() {
            Some("system") => {
                if !system.is_empty() && !text.is_empty() {
                    system.push('\n');
                }
                system.push_str(text);
                continue;
            }
            Some("assistant") => {
                let mut v = vec![];
                if let Some(p) = m["parts"].as_array() {
                    v = p.clone();
                    for item in &v {
                        if let Some(name) = item["functionCall"]["name"].as_str() {
                            wait.push_back(name.into());
                        }
                    }
                } else {
                    if !text.trim().is_empty() {
                        v.push(json!({"text": text}));
                    }
                    for c in m["tool_calls"].as_array().into_iter().flatten() {
                        let f = &c["function"];
                        let name = f["name"].as_str().unwrap_or("");
                        let args = match &f["arguments"] {
                            Value::Object(_) => f["arguments"].clone(),
                            Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                            _ => json!({}),
                        };
                        v.push(json!({"functionCall": {"name": name, "args": args}}));
                        wait.push_back(name.into());
                    }
                }
                ("model", v)
            }
            Some("tool") => {
                let Some(name) = wait.pop_front() else { continue };
                let resp = if text.starts_with("Error: ") {
                    json!({"error": text})
                } else {
                    json!({"result": text})
                };
                ("user", vec![json!({"functionResponse": {"name": name, "response": resp}})])
            }
            _ if text.trim().is_empty() => continue,
            _ => ("user", vec![json!({"text": text})]),
        };
        if parts.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some((r, p)) if *r == role => p.extend(parts),
            _ => out.push((role, parts)),
        }
    }
    let contents = out.into_iter().map(|(r, p)| json!({"role": r, "parts": p})).collect();
    (system, contents)
}

fn convert_tools(tools: &Value) -> Vec<Value> {
    let decls: Vec<Value> = tools
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            let f = &t["function"];
            json!({
                "name": f["name"],
                "description": f["description"],
                "parameters": f["parameters"],
            })
        })
        .collect();
    if decls.is_empty() { vec![] } else { vec![json!({"functionDeclarations": decls})] }
}

fn body(p: &Provider, msgs: &[Value], tools: &Value, mode: Mode) -> Value {
    let (system, contents) = convert(msgs);
    let mut cfg = json!({
        "maxOutputTokens": if mode == Mode::Turn { p.max_out } else { p.max_out.min(8_000) }
    });
    if let Some(t) = p.temperature {
        cfg["temperature"] = json!(t);
    }
    if p.json {
        cfg["responseMimeType"] = json!("application/json");
    }
    if p.no_think {
        cfg["thinkingConfig"] = json!({"thinkingBudget": 0});
    } else if p.think {
        cfg["thinkingConfig"] = json!({"includeThoughts": true});
    }
    let mut b = json!({
        "contents": contents,
        "generationConfig": cfg,
    });
    if !system.trim().is_empty() {
        b["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    if tools.is_array() {
        let t = convert_tools(tools);
        if !t.is_empty() {
            b["tools"] = json!(t);
            if mode == Mode::Summary {
                b["toolConfig"] = json!({"functionCallingConfig": {"mode": "NONE"}});
            }
        }
    }
    b
}

fn push_raw_part(raw: &mut Vec<Value>, part: &Value) {
    let is_plain_text = part.get("text").and_then(Value::as_str).is_some()
        && part.get("functionCall").is_none()
        && part.get("thoughtSignature").is_none();
    if is_plain_text
        && let Some(prev) = raw.last_mut()
        && prev.get("text").and_then(Value::as_str).is_some()
        && prev.get("thought") == part.get("thought")
        && prev.get("thoughtSignature").is_none()
    {
        let merged = format!("{}{}", prev["text"].as_str().unwrap_or(""), part["text"].as_str().unwrap_or(""));
        prev["text"] = json!(merged);
        return;
    }
    raw.push(part.clone());
}

pub fn chat(p: &Provider, msgs: &[Value], tools: &Value, mode: Mode, cancel: &Cancel, on: &mut dyn FnMut(Delta)) -> Result<Reply, Fail> {
    let body = body(p, msgs, tools, mode);
    let headers = p.headers()?;
    let model = p.model.strip_prefix("models/").unwrap_or(&p.model);
    let path = format!("/models/{model}:streamGenerateContent?alt=sse");
    let mut out = Reply::default();
    let mut raw_parts: Vec<Value> = vec![];
    let mut finish = String::new();
    let mut plain = String::new();
    let req = Req { base: &p.base, method: "POST", path: &path, headers: &headers, body: &body.to_string() };
    http::request(&req, cancel, &mut |l| {
        let Some(d) = sse(l) else {
            if plain.len() < 500 {
                plain.push_str(l);
            }
            return Ok(());
        };
        let v: Value = serde_json::from_str(d).map_err(|e| format!("JSON model: {e}"))?;
        if !v["error"].is_null() {
            return Err(v["error"]["message"].as_str().unwrap_or("kesalahan server").to_string());
        }
        let u = &v["usageMetadata"];
        if let Some(n) = u["totalTokenCount"].as_u64() {
            out.tokens = n;
        } else if u.is_object() {
            let sum = u["promptTokenCount"].as_u64().unwrap_or(0)
                + u["candidatesTokenCount"].as_u64().unwrap_or(0)
                + u["thoughtsTokenCount"].as_u64().unwrap_or(0);
            if sum > 0 {
                out.tokens = sum;
            }
        }
        let cand = &v["candidates"][0];
        for part in cand["content"]["parts"].as_array().into_iter().flatten() {
            push_raw_part(&mut raw_parts, part);
            if part["thought"].as_bool() == Some(true) {
                if let Some(s) = part["text"].as_str().filter(|s| !s.is_empty()) {
                    out.thinking.push_str(s);
                    on(Delta::Think);
                }
            } else if let Some(s) = part["text"].as_str().filter(|s| !s.is_empty()) {
                out.content.push_str(s);
                on(Delta::Text(s));
            }
            if let Some(fc) = part.get("functionCall").filter(|v| v.is_object()) {
                let name = fc["name"].as_str().unwrap_or("");
                let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                out.calls.push(json!({"function": {"name": name, "arguments": args}}));
            }
        }
        if let Some(f) = cand["finishReason"].as_str().filter(|s| !s.is_empty()) {
            finish = f.into();
        }
        Ok(())
    })?;
    if finish.is_empty() {
        let why = if plain.is_empty() {
            "koneksi terputus sebelum jawaban selesai".into()
        } else {
            format!("balasan tidak dikenal: {plain}")
        };
        return Err(Fail::from(why));
    }
    out.done_reason = match finish.as_str() {
        "MAX_TOKENS" => "length",
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            out.note = Some(finish.clone());
            "refusal"
        }
        _ => "stop",
    }.into();
    if !raw_parts.is_empty() {
        out.parts = Some(json!(raw_parts));
    }
    Ok(out)
}

pub fn models(p: &Provider, cancel: &Cancel) -> Result<Vec<String>, String> {
    let v = p.get_json("/models?pageSize=1000", cancel)?;
    let list = v["models"].as_array().cloned().unwrap_or_default();
    Ok(list
        .iter()
        .filter(|m| {
            m["supportedGenerationMethods"]
                .as_array()
                .is_none_or(|a| a.iter().any(|x| x == "generateContent"))
        })
        .filter_map(|m| m["name"].as_str().map(|s| s.strip_prefix("models/").unwrap_or(s).to_string()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn konversi_dan_gabung_tool() {
        let m = [
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"u"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"function":{"name":"read","arguments":{"path":"a"}}},
                {"function":{"name":"bash","arguments":"{}"}}
            ]}),
            json!({"role":"tool","content":"isi"}),
            json!({"role":"tool","content":"Error: gagal"}),
        ];
        let (sys, c) = convert(&m);
        assert_eq!(sys, "sys");
        assert_eq!(c.len(), 3);
        assert_eq!(c[1]["role"], "model");
        assert_eq!(c[1]["parts"].as_array().unwrap().len(), 2);
        assert_eq!(c[2]["role"], "user");
        assert_eq!(c[2]["parts"][0]["functionResponse"]["name"], "read");
        assert_eq!(c[2]["parts"][1]["functionResponse"]["response"]["error"], "Error: gagal");
    }

    #[test]
    fn gabung_part_teks() {
        let mut raw = vec![];
        push_raw_part(&mut raw, &json!({"text": "ha"}));
        push_raw_part(&mut raw, &json!({"text": "lo"}));
        push_raw_part(&mut raw, &json!({"functionCall": {"name": "read", "args": {}}, "thoughtSignature": "SIG"}));
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0]["text"], "halo");
        assert_eq!(raw[1]["thoughtSignature"], "SIG");
    }
}
