//! Tool lewat teks (`--tools text`), untuk server yang tidak meneruskan `tools`.
//! Riwayat netral diubah menjadi pesan biasa, dan permintaan tool dibaca dari jawaban.
//!
//! Blok dikenali hanya di awal baris dan di luar blok kode. Sesudah blok pertama hanya blok
//! lain yang dibaca; sisanya dibuang karena model kecil suka mengarang hasil tool.

use crate::provider::{Delta, Mode, Reply};
use serde_json::{Value, json};

const OPEN: &str = "<use_tool";
const CLOSE: &str = "</use_tool>";
/// Hanya ditulis llmy. Bila model menulisnya, itu karangan: jawaban dipotong di sana.
const FAKE: &str = "<tool_result";

const RULES: &str = r#"

# Tools
To call a tool, write this block at the start of a line, outside any code fence:
<use_tool name="read">
{"path": "."}
</use_tool>
The tag is always use_tool. The body is one JSON object matching the tool's parameters, with newlines in strings written as \n. Write nothing after the last block: the result arrives in the next message as <tool_result>. Never write <tool_result> yourself, and never claim a tool ran before you have its result. If no tool is needed, just answer.

Available tools:
"#;

/// Riwayat netral -> system/user/assistant berisi teks saja. Role beruntun digabung.
pub fn convert(msgs: &[Value], tools: &Value, mode: Mode) -> Vec<Value> {
    let mut out: Vec<(&str, String)> = vec![];
    for m in msgs {
        let text = m["content"].as_str().unwrap_or("");
        let (role, text) = match m["role"].as_str() {
            Some("system") => {
                let mut s = text.to_string();
                // Saat meringkas tetap disertakan: riwayatnya memuat blok yang sama.
                s += RULES;
                for t in tools.as_array().into_iter().flatten() {
                    s += &format!("{}\n", t["function"]);
                }
                ("system", s)
            }
            Some("assistant") => {
                let mut s = text.to_string();
                for c in m["tool_calls"].as_array().into_iter().flatten() {
                    let f = &c["function"];
                    let args = f["arguments"].as_str().map_or_else(|| f["arguments"].to_string(), String::from);
                    if !s.is_empty() {
                        s += "\n";
                    }
                    s += &format!("{OPEN} name=\"{}\">\n{args}\n{CLOSE}", f["name"].as_str().unwrap_or(""));
                }
                ("assistant", s)
            }
            Some("tool") => {
                // Isi file tidak boleh menutup pembungkusnya.
                let t = text.replace("</tool_result", "<\\/tool_result");
                let name = m["tool_name"].as_str().unwrap_or("");
                ("user", format!("<tool_result name=\"{name}\">\n{t}\n</tool_result>"))
            }
            _ => ("user", text.to_string()),
        };
        if text.trim().is_empty() {
            continue;
        }
        match out.last_mut() {
            Some((r, s)) if *r == role => {
                s.push_str("\n\n");
                s.push_str(&text);
            }
            _ => out.push((role, text)),
        }
    }
    if mode == Mode::Summary
        && let Some((_, s)) = out.last_mut()
    {
        s.push_str(" Do not write <use_tool>.");
    }
    out.into_iter().map(|(r, t)| json!({"role":r,"content":t})).collect()
}

/// Menahan blok tool agar tidak tampil di layar selagi jawaban mengalir.
#[derive(Default)]
pub struct Filter {
    all: String,
    shown: usize,
    /// Awal baris yang sedang dibaca.
    line: usize,
    fence: Option<String>,
    held: bool,
}

impl Filter {
    pub fn push(&mut self, s: &str, on: &mut dyn FnMut(Delta)) {
        self.all.push_str(s);
        while !self.held {
            let rest = &self.all[self.line..];
            let end = rest.find('\n');
            let l = &rest[..end.unwrap_or(rest.len())];
            let t = l.trim_start_matches([' ', '\t']);
            if self.fence.is_none() {
                if is_open(t) || t.starts_with(FAKE) {
                    self.held = true;
                    break;
                }
                if end.is_none() && (OPEN.starts_with(t) || FAKE.starts_with(t)) {
                    break; // belum pasti
                }
            }
            match end {
                Some(i) => {
                    fence(l, &mut self.fence);
                    self.line += i + 1;
                    self.show(self.line, on);
                }
                None => {
                    self.show(self.all.len(), on);
                    break;
                }
            }
        }
    }

    fn show(&mut self, to: usize, on: &mut dyn FnMut(Delta)) {
        if to > self.shown {
            on(Delta::Text(&self.all[self.shown..to]));
            self.shown = to;
        }
    }

    /// Pindahkan permintaan tool dari teks ke `r.calls`; teks yang ternyata bukan
    /// permintaan ditampilkan.
    pub fn finish(mut self, r: &mut Reply, on: &mut dyn FnMut(Delta)) {
        r.blocks = None;
        r.thinking.clear();
        r.reasoning.clear();
        if matches!(r.done_reason.as_str(), "length" | "refusal") {
            // Terpotong: blok yang tertahan mungkin separuh, jangan ditampilkan atau dijalankan.
            r.content = self.all[..self.line.max(self.shown).min(self.all.len())].trim_end().into();
            return;
        }
        let (text, calls) = parse(&self.all);
        r.content = text.trim_end().into();
        let end = text.len();
        if calls.is_empty() {
            self.show(end, on);
            // Sebagian server tetap mengurai panggilan native, dengan nama berawalan.
            for c in &mut r.calls {
                let n = c["function"]["name"].as_str().unwrap_or("");
                let n = n.rsplit([':', '.', '/']).next().unwrap_or(n).to_string();
                c["function"]["name"] = json!(n);
            }
        } else {
            r.calls = calls;
        }
    }
}

fn is_open(t: &str) -> bool {
    t.strip_prefix(OPEN).is_some_and(|r| r.starts_with([' ', '\t', '>']))
}

/// Perbarui status blok kode menurut satu baris.
fn fence(l: &str, cur: &mut Option<String>) {
    let t = l.trim_start_matches(' ');
    if l.len() - t.len() > 3 {
        return;
    }
    let Some(c) = t.chars().next().filter(|c| matches!(c, '`' | '~')) else { return };
    let n = t.chars().take_while(|x| *x == c).count();
    if n < 3 {
        return;
    }
    match cur {
        None => *cur = Some(t[..n].into()),
        Some(f) if t.starts_with(f.as_str()) && t[n..].trim().is_empty() => *cur = None,
        Some(_) => {}
    }
}

fn parse(s: &str) -> (&str, Vec<Value>) {
    let mut f = None;
    let mut pos = 0;
    for l in s.split_inclusive('\n') {
        if f.is_none()
            && let Some((c, n)) = one(&s[pos..])
        {
            let mut calls = vec![c];
            let mut rest = &s[pos + n..];
            loop {
                let t = rest.trim_start();
                let Some((c, n)) = one(t) else { return (&s[..pos], calls) };
                calls.push(c);
                rest = &t[n..];
            }
        }
        if f.is_none() && l.trim_start_matches([' ', '\t']).starts_with(FAKE) {
            return (&s[..pos], vec![]);
        }
        fence(l, &mut f);
        pos += l.len();
    }
    (s, vec![])
}

/// Satu blok di awal `s`, beserta panjangnya. Nama tool tak dikenal tetap diterima:
/// pemanggil menolaknya dengan pesan yang bisa diperbaiki model.
fn one(s: &str) -> Option<(Value, usize)> {
    let t = s.trim_start_matches([' ', '\t']);
    if !is_open(t) {
        return None;
    }
    let gt = t.find('>')?;
    let attr = t[OPEN.len()..gt].trim();
    let mut name = attr.strip_prefix("name=").map_or("", |n| n.trim_matches(['"', '\''])).to_string();
    let body = &t[gt + 1..];
    let before = s.len() - body.len();
    // JSON dibaca lebih dulu: isi argumen boleh memuat teks penutup blok.
    let mut it = serde_json::Deserializer::from_str(body).into_iter::<Value>();
    let (mut args, used) = match it.next() {
        Some(Ok(v)) if v.is_object() => {
            let n = it.byte_offset();
            let rest = &body[n..];
            let skip = rest.len() - rest.trim_start().len();
            (v, if rest[skip..].starts_with(CLOSE) { n + skip + CLOSE.len() } else { n })
        }
        _ => {
            // Tanpa penutup berarti terpotong.
            let i = body.find(CLOSE)?;
            let raw = body[..i].trim();
            let v = if raw.is_empty() { json!({}) } else { Value::String(raw.into()) };
            (v, i + CLOSE.len())
        }
    };
    // Bentuk lain yang sering ditulis model: {"name": ..., "arguments": {...}}.
    if args["arguments"].is_object()
        && let Some(n) = args["name"].as_str()
        && (name.is_empty() || name == n)
    {
        name = n.into();
        args = args["arguments"].take();
    }
    if name.is_empty() {
        return None;
    }
    Some((json!({"function":{"name":name,"arguments":args}}), before + used))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str], done: &str) -> (String, Reply) {
        let mut shown = String::new();
        let mut f = Filter::default();
        let mut r = Reply { done_reason: done.into(), content: chunks.concat(), ..Default::default() };
        let mut on = |d: Delta| {
            if let Delta::Text(s) = d {
                shown.push_str(s);
            }
        };
        for c in chunks {
            f.push(c, &mut on);
        }
        f.finish(&mut r, &mut on);
        (shown, r)
    }

    fn calls(r: &Reply) -> Vec<String> {
        r.calls.iter().map(|c| format!("{} {}", c["function"]["name"].as_str().unwrap(), c["function"]["arguments"])).collect()
    }

    #[test]
    fn membaca_blok() {
        let cases: &[(&str, &str, &[&str])] = &[
            ("halo\ndunia", "halo\ndunia", &[]),
            ("Baca.\n<use_tool name=\"read\">\n{\"path\": \"a\"}\n</use_tool>\n", "Baca.", &["read {\"path\":\"a\"}"]),
            ("<use_tool name='read'>{\"path\":\"a\"}</use_tool>\n<use_tool name=\"grep\">\n{}\n</use_tool>\n<tool_result>palsu</tool_result>", "",
                &["read {\"path\":\"a\"}", "grep {}"]),
            ("<use_tool name=\"write\">\n{\"content\":\"</use_tool>\\n\"}\n</use_tool>", "", &["write {\"content\":\"</use_tool>\\n\"}"]),
            ("<use_tool name=\"read\">\n{path: a}\n</use_tool>", "", &["read \"{path: a}\""]),
            ("<use_tool name=\"read\">\n</use_tool>", "", &["read {}"]),
            ("<use_tool name=\"read\">\n{\"path\":\"a\"}", "", &["read {\"path\":\"a\"}"]),
            ("<use_tool>\n{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}\n</use_tool>", "", &["read {\"path\":\"a\"}"]),
            ("x\n<use_tool name=\"read\">\n{\"path", "x\n<use_tool name=\"read\">\n{\"path", &[]),
            ("lihat <use_tool name=\"read\">{}</use_tool>", "lihat <use_tool name=\"read\">{}</use_tool>", &[]),
            ("<use_tools>\n<use", "<use_tools>\n<use", &[]),
            ("```xml\n<use_tool name=\"read\">\n{}\n</use_tool>\n```", "```xml\n<use_tool name=\"read\">\n{}\n</use_tool>\n```", &[]),
            ("Baca.\n<tool_result name=\"read\">\nkarangan\n</tool_result>", "Baca.", &[]),
            ("```\nx\n```\n<use_tool name=\"read\">{}</use_tool>", "```\nx\n```", &["read {}"]),
        ];
        for (input, text, want) in cases {
            // Hasilnya tidak boleh bergantung pada cara potongan datang.
            let bytes: Vec<String> = input.chars().map(String::from).collect();
            let each: Vec<&str> = bytes.iter().map(String::as_str).collect();
            for chunks in [vec![*input], each] {
                let (shown, r) = run(&chunks, "stop");
                assert_eq!(shown.trim_end(), *text, "{input}");
                assert_eq!(r.content.trim_end(), *text, "{input}");
                assert_eq!(calls(&r), *want, "{input}");
            }
        }
    }

    #[test]
    fn terpotong_tidak_dijalankan() {
        let (shown, r) = run(&["Baca.\n<use_tool name=\"bash\">\n{\"command\":\"rm x\"}", "\n</use_tool>"], "length");
        assert_eq!((shown.as_str(), r.content.as_str()), ("Baca.\n", "Baca."));
        assert!(r.calls.is_empty());
    }

    #[test]
    fn konversi() {
        let tools = json!([{"type":"function","function":{"name":"read"}}]);
        let m = [
            json!({"role":"system","content":"S"}),
            json!({"role":"user","content":"u"}),
            json!({"role":"assistant","content":"","blocks":[1],"thinking":"t","tool_calls":[
                {"id":"x","function":{"name":"read","arguments":{"path":"a"}}},{"function":{"name":"read","arguments":"rusak"}}]}),
            json!({"role":"tool","tool_name":"read","content":"isi </tool_result> jahat"}),
            json!({"role":"tool","tool_name":"read","content":"dua"}),
            json!({"role":"user","content":"lanjut"}),
        ];
        let o = convert(&m, &tools, Mode::Turn);
        let roles: Vec<&str> = o.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert!(o[0]["content"].as_str().unwrap().ends_with("{\"name\":\"read\"}\n"));
        assert!(o.iter().all(|m| m.as_object().unwrap().len() == 2));
        let back = run(&[o[2]["content"].as_str().unwrap()], "stop").1;
        assert_eq!(calls(&back), ["read {\"path\":\"a\"}", "read \"rusak\""]);
        let t = o[3]["content"].as_str().unwrap();
        assert!(t.matches("</tool_result>").count() == 2 && t.ends_with("lanjut"));
    }
}
