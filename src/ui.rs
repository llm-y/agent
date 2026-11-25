//! Antarmuka inline: teks selesai dicetak biasa (scroll & salin terminal tetap
//! jalan), hanya area bawah yang digambar ulang, dan hanya bila isinya berubah.

use crate::agent::{Cmd, Ev};
use crate::cancel::Cancel;
use crate::editor::Editor;
use crate::perm::Answer;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::{event, execute, terminal};
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthChar;

const RST: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const SPIN: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// Tombol diabaikan sejenak saat menu izin muncul, agar ketikan yang sedang
/// berjalan tidak ikut menyetujui.
const ASK_GUARD: Duration = Duration::from_millis(500);

const COMMANDS: [(&str, &str); 10] = [
    ("/help", "bantuan"),
    ("/model", "lihat atau ganti model"),
    ("/api", "atur atau pilih API provider"),
    ("/think", "aktif/nonaktifkan thinking"),
    ("/compact", "ringkas percakapan untuk menghemat context"),
    ("/remember", "simpan fakta: /remember <fakta> atau /remember all <fakta>"),
    ("/memory", "lihat memori proyek dan semua proyek"),
    ("/clear", "mulai percakapan baru"),
    ("/copy", "salin jawaban terakhir (OSC 52)"),
    ("/exit", "keluar"),
];

/// Buang karakter kontrol dan pembalik arah teks: isi dari model, file, atau
/// perintah tidak boleh mengendalikan terminal.
pub fn sanitize(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => o.push(c),
            '\t' => o.push_str("    "),
            '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' => o.push('\u{fffd}'),
            c if c.is_control() => {}
            c => o.push(c),
        }
    }
    o
}

fn cw(c: char) -> usize {
    c.width().unwrap_or(0)
}

fn sw(s: &str) -> usize {
    s.chars().map(cw).sum()
}

#[derive(Clone, PartialEq, Default)]
struct Row {
    s: String,
    w: usize,
}

impl Row {
    fn new(style: &str, text: &str, max: usize) -> Row {
        let (mut t, mut w) = (String::new(), 0);
        for c in text.chars() {
            if w + cw(c) > max {
                break;
            }
            t.push(c);
            w += cw(c);
        }
        let s = if style.is_empty() { t } else { format!("{style}{t}{RST}") };
        Row { s, w }
    }
}

type Span = (&'static str, String);

/// Bungkus per kata. Teks span harus sudah bersih dari karakter kontrol.
fn wrap(spans: &[Span], width: usize, first: Span, rest: &str) -> Vec<Row> {
    let indent = sw(&first.1).max(sw(rest));
    let max = width.saturating_sub(indent).max(4);
    let cells: Vec<(char, &'static str)> =
        spans.iter().flat_map(|(st, t)| t.chars().map(move |c| (c, *st))).collect();
    let mut rows = vec![];
    let mut i = 0;
    loop {
        let (mut j, mut w, mut brk) = (i, 0, None);
        while j < cells.len() && w + cw(cells[j].0) <= max {
            w += cw(cells[j].0);
            j += 1;
            if cells[j - 1].0 == ' ' {
                brk = Some(j);
            }
        }
        if j < cells.len() {
            if cells[j].0 != ' '
                && let Some(b) = brk
            {
                j = b;
            }
            j = j.max(i + 1);
        }
        let mut s = if rows.is_empty() {
            if first.0.is_empty() { first.1.clone() } else { format!("{}{}{RST}", first.0, first.1) }
        } else {
            rest.to_string()
        };
        let mut w = if rows.is_empty() { sw(&first.1) } else { sw(rest) };
        let mut style = "";
        for (c, st) in &cells[i..j] {
            if *st != style {
                if !style.is_empty() {
                    s.push_str(RST);
                }
                s.push_str(st);
                style = st;
            }
            s.push(*c);
            w += cw(*c);
        }
        if !style.is_empty() {
            s.push_str(RST);
        }
        rows.push(Row { s, w });
        i = j;
        if i < cells.len() && cells[i].0 == ' ' {
            i += 1;
        }
        if i >= cells.len() {
            return rows;
        }
    }
}

/// Markdown ringan per baris: judul, pagar kode, `kode`, **tebal**.
fn markdown(line: &str, in_code: &mut bool) -> Vec<Span> {
    let t = line.trim_start();
    if t.starts_with("```") {
        *in_code = !*in_code;
        return vec![(DIM, line.into())];
    }
    if *in_code {
        return vec![(CYAN, line.into())];
    }
    if t.starts_with('#') && t.trim_start_matches('#').starts_with(' ') {
        return vec![(BOLD, t.trim_start_matches('#').trim().into())];
    }
    let mut v: Vec<Span> = vec![];
    let mut rest = line;
    while let Some(i) = rest.find(['`', '*']) {
        let (mark, style) = if rest[i..].starts_with("**") {
            ("**", BOLD)
        } else if rest[i..].starts_with('`') {
            ("`", CYAN)
        } else {
            ("", "")
        };
        let close = if mark.is_empty() { None } else { rest[i + mark.len()..].find(mark).filter(|n| *n > 0) };
        match close {
            Some(n) => {
                v.push(("", rest[..i].into()));
                let a = i + mark.len();
                v.push((style, rest[a..a + n].into()));
                rest = &rest[a + n + mark.len()..];
            }
            None => {
                v.push(("", rest[..i + mark.len().max(1)].into()));
                rest = &rest[i + mark.len().max(1)..];
            }
        }
    }
    v.push(("", rest.into()));
    v
}

/// Sebagian pengelola sesi (mis. `s`) melaporkan 0x0 sebelum klien tersambung.
fn size((w, h): (u16, u16)) -> (usize, usize) {
    let or = |n: u16, d: usize| if n == 0 { d } else { n as usize };
    (or(w, 80).max(20), or(h, 24).max(10))
}

struct Ask {
    question: String,
    always: Option<&'static str>,
    reply: Sender<Answer>,
    sel: usize,
    since: Instant,
}

impl Ask {
    fn options(&self) -> Vec<(Answer, &'static str)> {
        let mut v = vec![(Answer::Yes, "Ya")];
        if let Some(a) = self.always {
            v.push((Answer::Always, a));
        }
        v.push((Answer::No, "Tidak, beri arahan lain"));
        v
    }
}

struct Ui {
    w: usize,
    h: usize,
    live: Vec<Row>,
    cur: (usize, usize),
    drawn: bool,
    out: String,

    ed: Editor,
    busy: bool,
    started: Instant,
    queue: VecDeque<String>,
    ask: Option<Ask>,
    popup: Vec<(String, String)>,
    popup_sel: usize,
    popup_off: bool,
    files: Option<Vec<String>>,
    exit_hint: Option<Instant>,

    partial: String,
    in_text: bool,
    in_code: bool,
    answer: String,
    last_answer: String,
    thought: usize,
    tool: Option<String>,
    status: (u64, u64, String, bool),

    cwd: PathBuf,
    cancel: Arc<Cancel>,
    cmd: Sender<Cmd>,
    quit: bool,
}

pub fn restore() {
    let _ = execute!(io::stdout(), event::DisableBracketedPaste);
    let _ = terminal::disable_raw_mode();
    print!("{RST}\x1b[?25h\r\n");
    let _ = io::stdout().flush();
}

pub fn run(cwd: PathBuf, cancel: Arc<Cancel>, tx: Sender<Ev>, rx: Receiver<Ev>, cmd: Sender<Cmd>) -> io::Result<()> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |i| {
        restore();
        hook(i);
    }));
    terminal::enable_raw_mode()?;
    // Di konsol Windows lama bisa tidak didukung; tempel tetap jalan sebagai ketikan.
    let _ = execute!(io::stdout(), event::EnableBracketedPaste);
    std::thread::spawn(move || {
        while let Ok(e) = event::read() {
            if tx.send(Ev::Term(e)).is_err() {
                return;
            }
        }
        let _ = tx.send(Ev::TermClosed);
    });
    let (w, h) = size(terminal::size().unwrap_or((0, 0)));
    let mut ui = Ui {
        w, h, live: vec![], cur: (0, 0), drawn: false, out: String::new(),
        ed: Editor::default(), busy: false, started: Instant::now(), queue: VecDeque::new(), ask: None,
        popup: vec![], popup_sel: 0, popup_off: false, files: None, exit_hint: None,
        partial: String::new(), in_text: false, in_code: false, answer: String::new(), last_answer: String::new(),
        thought: 0, tool: None, status: (0, 1, String::new(), false),
        cwd, cancel: cancel.clone(), cmd, quit: false,
    };
    ui.intro();
    let r = ui.main_loop(rx);
    cancel.cancel();
    ui.erase();
    let r = r.and(ui.flush());
    restore();
    r
}

impl Ui {
    fn main_loop(&mut self, rx: Receiver<Ev>) -> io::Result<()> {
        self.render()?;
        while !self.quit {
            // Spinner 4x per detik saja: `s` mencatat setiap byte ke log.
            let tick = if self.busy || self.exit_hint.is_some() { 250 } else { 3_600_000 };
            match rx.recv_timeout(Duration::from_millis(tick)) {
                Ok(e) => {
                    self.event(e);
                    while let Ok(e) = rx.try_recv() {
                        self.event(e);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            if self.exit_hint.is_some_and(|t| t.elapsed() > Duration::from_secs(2)) {
                self.exit_hint = None;
            }
            self.render()?;
        }
        Ok(())
    }

    // ---- gambar ----

    /// Hapus area bawah; setelah ini kursor di kolom 0 tempat area itu mulai.
    fn erase(&mut self) {
        if !self.drawn {
            return;
        }
        // Setelah terminal menyempit, baris lama bisa membungkus jadi beberapa baris.
        let up: usize = self.live[..self.cur.0].iter().map(|r| r.w.saturating_sub(1) / self.w).sum::<usize>()
            + self.cur.0 + self.cur.1 / self.w;
        self.out.push('\r');
        if up > 0 {
            self.out += &format!("\x1b[{up}A");
        }
        self.out += "\x1b[J";
        self.drawn = false;
    }

    /// Cetak permanen ke atas area bawah.
    fn print(&mut self, rows: Vec<Row>) {
        self.erase();
        for r in rows {
            self.out += &r.s;
            self.out += "\r\n";
        }
    }

    fn blank(&mut self) {
        self.print(vec![Row::default()]);
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.out.is_empty() {
            return Ok(());
        }
        let s = format!("\x1b[?2026h{}\x1b[?2026l", self.out);
        self.out.clear();
        let mut o = io::stdout().lock();
        o.write_all(s.as_bytes())?;
        o.flush()
    }

    fn render(&mut self) -> io::Result<()> {
        let (rows, cur) = self.compose();
        if self.drawn && rows == self.live && cur == self.cur {
            return self.flush();
        }
        self.erase();
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                self.out += "\r\n";
            }
            self.out += &r.s;
        }
        let up = rows.len() - 1 - cur.0;
        self.out.push('\r');
        if up > 0 {
            self.out += &format!("\x1b[{up}A");
        }
        if cur.1 > 0 {
            self.out += &format!("\x1b[{}C", cur.1);
        }
        self.live = rows;
        self.cur = cur;
        self.drawn = true;
        self.flush()
    }

    fn compose(&self) -> (Vec<Row>, (usize, usize)) {
        let w = self.w - 1;
        let mut top: Vec<Row> = vec![];
        if !self.partial.is_empty() {
            let mut code = self.in_code;
            let first = if self.in_text { "  " } else { "● " };
            top.extend(wrap(&markdown(&self.partial, &mut code), w, ("", first.into()), "  "));
        }
        if self.busy && self.ask.is_none() {
            top.push(Row::default());
            let t = self.started.elapsed();
            let spin = SPIN[(t.as_millis() / 250) as usize % SPIN.len()];
            let what = match &self.tool {
                Some(t) => t.clone(),
                None if self.thought > 0 && self.answer.is_empty() => format!("Berpikir ({} token)", self.thought),
                None => "Bekerja".into(),
            };
            top.push(Row::new(YELLOW, &format!("{spin} {what}… {} dtk · Esc untuk berhenti", t.as_secs()), w));
        }
        for q in self.queue.iter().take(3) {
            top.push(Row::new(DIM, &format!("  antre: {}", q.replace('\n', " ")), w));
        }

        let (mut bottom, cur) = if let Some(a) = &self.ask {
            let mut v = vec![Row::default()];
            v.extend(wrap(&[(BOLD, sanitize(&a.question))], w, ("", "  ".into()), "  "));
            let mut at = 0;
            for (i, (_, l)) in a.options().iter().enumerate() {
                let on = i == a.sel;
                if on {
                    at = v.len();
                }
                let m = if on { "›" } else { " " };
                v.push(Row::new(if on { CYAN } else { "" }, &format!("  {m} {}. {l}", i + 1), w));
            }
            v.push(Row::new(DIM, "  ↑/↓ pilih · Enter · Esc = tidak", w));
            (v, (at, 2))
        } else {
            let (mut v, cur) = self.input_box(w);
            if self.popup.is_empty() {
                v.push(self.footer(w));
            } else {
                let from = self.popup_sel.saturating_sub(7);
                for (i, (k, d)) in self.popup.iter().enumerate().skip(from).take(COMMANDS.len()) {
                    let on = i == self.popup_sel;
                    let m = if on { "›" } else { " " };
                    v.push(Row::new(if on { CYAN } else { DIM }, &format!(" {m} {k}  {d}"), w));
                }
            }
            (v, cur)
        };

        // Area bawah tak boleh melebihi layar: baris yang tergulung tak bisa dihapus.
        let room = (self.h - 1).saturating_sub(bottom.len());
        if top.len() > room {
            top.drain(..top.len() - room);
        }
        let n = top.len();
        top.append(&mut bottom);
        (top, (cur.0 + n, cur.1))
    }

    fn input_box(&self, w: usize) -> (Vec<Row>, (usize, usize)) {
        let inner = w - 6;
        // Pecah jadi baris visual sambil melacak kursor.
        let mut lines = vec![(String::new(), 0usize)];
        let mut cur = (0, 0);
        for (i, &c) in self.ed.buf.iter().enumerate() {
            if c != '\n' && lines.last().unwrap().1 + cw(c) > inner {
                lines.push((String::new(), 0));
            }
            if i == self.ed.cur {
                cur = (lines.len() - 1, lines.last().unwrap().1);
            }
            if c == '\n' {
                lines.push((String::new(), 0));
            } else {
                let l = lines.last_mut().unwrap();
                l.0.push(c);
                l.1 += cw(c);
            }
        }
        if self.ed.cur == self.ed.buf.len() {
            cur = (lines.len() - 1, lines.last().unwrap().1);
        }
        let vis = 8.min(self.h.saturating_sub(6)).max(1);
        let from = (cur.0 + 1).saturating_sub(vis);
        let mut v = vec![Row::new(DIM, &format!("╭{}╮", "─".repeat(w - 2)), w)];
        for (i, (l, lw)) in lines.iter().enumerate().skip(from).take(vis) {
            let p = if i == 0 { "›" } else { " " };
            let pad = " ".repeat(inner - lw.min(&inner) + 1);
            v.push(Row { s: format!("{DIM}│{RST} {p} {l}{pad}{DIM}│{RST}"), w });
        }
        v.push(Row::new(DIM, &format!("╰{}╯", "─".repeat(w - 2)), w));
        (v, (1 + cur.0 - from, 4 + cur.1))
    }

    fn footer(&self, w: usize) -> Row {
        if self.exit_hint.is_some() {
            return Row::new(YELLOW, "  Ctrl-C sekali lagi untuk keluar", w);
        }
        let k = |n: u64| if n < 1000 { n.to_string() } else { format!("{:.1}k", n as f64 / 1000.0) };
        let (t, c, m, th) = &self.status;
        let th = if *th { " · thinking" } else { "" };
        Row::new(DIM, &sanitize(&format!("  {m}{th} · {}/{} · /help", k(*t), k(*c))), w)
    }

    // ---- isi ----

    fn intro(&mut self) {
        let w = self.w - 1;
        let v = crate::VERSION;
        // `\\?\` dari canonicalize di Windows tidak berguna bagi pembaca.
        let dir = sanitize(self.cwd.display().to_string().trim_start_matches(r"\\?\"));
        self.print(vec![
            Row::default(),
            Row::new(BOLD, &format!("  llmy {v}"), w),
            Row::new(DIM, &format!("  {dir}"), w),
        ]);
        self.blank();
    }

    fn lines(&mut self, style: &'static str, first: &str, text: &str) {
        let mut rows = vec![];
        for (i, l) in sanitize(text).lines().enumerate() {
            let f = if i == 0 { first.to_string() } else { " ".repeat(sw(first)) };
            rows.extend(wrap(&[(style, l.into())], self.w - 1, (style, f), &" ".repeat(sw(first))));
        }
        self.print(rows);
        self.blank();
    }

    fn text(&mut self, s: &str) {
        self.answer.push_str(s);
        self.partial.push_str(&sanitize(s));
        if !self.in_text {
            let t = self.partial.trim_start().to_string();
            self.partial = t;
        }
        while let Some(i) = self.partial.find('\n') {
            let l: String = self.partial.drain(..=i).collect();
            self.commit(l.trim_end());
        }
    }

    fn commit(&mut self, l: &str) {
        let first = if self.in_text { "  " } else { "● " };
        self.in_text = true;
        let spans = markdown(l, &mut self.in_code);
        let rows = wrap(&spans, self.w - 1, ("", first.into()), "  ");
        self.print(rows);
    }

    fn end_text(&mut self) {
        let l = std::mem::take(&mut self.partial);
        if !l.trim().is_empty() {
            self.commit(l.trim_end());
        }
        if self.in_text {
            self.blank();
        }
        if !self.answer.trim().is_empty() {
            self.last_answer = std::mem::take(&mut self.answer);
        }
        self.answer.clear();
        self.in_text = false;
        self.in_code = false;
        self.thought = 0;
    }

    fn event(&mut self, e: Ev) {
        match e {
            Ev::Term(Event::Key(k)) if k.kind != KeyEventKind::Release => self.key(k),
            Ev::Term(Event::Paste(s)) if self.ask.is_none() => {
                self.ed.insert(&s);
                self.refresh_popup();
            }
            Ev::Term(Event::Resize(w, h)) => {
                (self.w, self.h) = size((w, h));
                self.erase();
            }
            Ev::Term(_) => {}
            Ev::TermClosed => self.quit = true,
            Ev::Text(s) => self.text(&s),
            Ev::Think(n) => self.thought = n,
            Ev::User(s) => self.lines(DIM, "› ", &s),
            Ev::EndText => self.end_text(),
            Ev::Work(s) => self.tool = Some(s.into()),
            Ev::Tool { title, detail } => {
                self.end_text();
                let title = sanitize(&title);
                let w = self.w - 1;
                let mut rows = wrap(&[(BOLD, title.clone())], w, (CYAN, "● ".into()), "  ");
                for (c, n, l) in detail {
                    let st = match c {
                        '-' => RED,
                        '+' => GREEN,
                        '$' => "",
                        _ => DIM,
                    };
                    let f = if n > 0 { format!("  {n:>4} {c} ") } else { format!("  {c} ") };
                    let rest = " ".repeat(sw(&f));
                    rows.extend(wrap(&[(st, sanitize(&l))], w, (DIM, f), &rest));
                }
                self.print(rows);
                self.tool = Some(title);
                self.files = None;
            }
            Ev::ToolDone { ok, summary } => {
                self.tool = None;
                let st = if ok { DIM } else { RED };
                let first = if ok { "  └ " } else { "  ✗ " };
                self.lines(st, first, &summary);
            }
            Ev::Ask { question, always, reply } => {
                self.ask = Some(Ask { question, always, reply, sel: 0, since: Instant::now() });
            }
            Ev::Status { tokens, ctx, model, think } => self.status = (tokens, ctx, model, think),
            Ev::Info(s) => {
                self.end_text();
                self.tool = None;
                self.lines(DIM, "  ", &s);
            }
            Ev::Error(s) => {
                self.end_text();
                self.lines(RED, "✗ ", &s);
            }
            Ev::Done => {
                self.end_text();
                self.busy = false;
                self.tool = None;
                if let Some(q) = self.queue.pop_front() {
                    self.submit(q);
                }
            }
        }
    }

    // ---- input ----

    fn start(&mut self, c: Cmd) {
        // Reset di sini, bukan di thread agent, supaya Esc tepat setelah Enter tidak hilang.
        self.cancel.reset();
        self.busy = true;
        self.started = Instant::now();
        let _ = self.cmd.send(c);
    }

    fn submit(&mut self, s: String) {
        let t = s.trim().to_string();
        if t.is_empty() {
            return;
        }
        if self.busy {
            return self.queue.push_back(t);
        }
        let shown = if t.starts_with("/api ") { crate::api::mask_command_display(&t) } else { t.clone() };
        self.lines(DIM, "› ", &shown);
        // Hanya kata tanpa '/' kedua dianggap perintah, jadi path absolut tetap pesan biasa.
        let (name, arg) = t.split_once(char::is_whitespace).unwrap_or((&t, ""));
        if !name.starts_with('/') || name[1..].contains('/') {
            return self.start(Cmd::User(t));
        }
        match name {
            "/exit" | "/quit" => self.quit = true,
            "/clear" => self.start(Cmd::Clear),
            "/compact" => self.start(Cmd::Compact),
            "/think" => self.start(Cmd::Think),
            "/remember" if arg.trim().is_empty() => self.lines(DIM, "  ", "Pakai: /remember <fakta> (proyek ini) atau /remember all <fakta> (semua proyek)"),
            "/remember" => self.start(Cmd::Remember(arg.trim().into())),
            "/memory" => self.start(Cmd::Memory),
            "/model" => {
                let a = arg.trim();
                self.start(Cmd::Model((!a.is_empty()).then(|| a.to_string())));
            }
            "/api" => self.start(Cmd::Api(arg.trim().into())),
            "/copy" => {
                let b = base64(self.last_answer.trim().as_bytes());
                self.erase();
                self.out += &format!("\x1b]52;c;{b}\x07");
                self.lines(DIM, "  ", "Jawaban terakhir dikirim ke clipboard terminal.");
            }
            "/help" => {
                let mut h = String::new();
                for (k, d) in COMMANDS {
                    h += &format!("{k:<10} {d}\n");
                }
                h += "\n@        sebut file\nEnter    kirim (saat bekerja: antre)\nAlt+Enter, Ctrl+J, atau \\ lalu Enter: baris baru\nEsc      hentikan giliran\nCtrl-C   hentikan; dua kali untuk keluar";
                self.lines(DIM, "  ", &h);
            }
            _ => self.lines(RED, "✗ ", &format!("Perintah {name} tidak ada. Lihat /help")),
        }
    }

    fn stop(&mut self) {
        // Pesan antre dikembalikan ke input, bukan langsung dijalankan.
        if !self.queue.is_empty() {
            let mut all: Vec<String> = self.queue.drain(..).collect();
            let cur = self.ed.text();
            if !cur.trim().is_empty() {
                all.push(cur);
            }
            self.ed.set(&all.join("\n"));
        }
        self.cancel.cancel();
    }

    fn answer_ask(&mut self, a: Answer) {
        if let Some(q) = self.ask.take() {
            if a == Answer::No {
                self.stop();
            }
            let _ = q.reply.send(a);
        }
    }

    fn key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        if let Some(a) = &mut self.ask {
            let is_stop = k.code == KeyCode::Esc || (ctrl && k.code == KeyCode::Char('c'));
            if a.since.elapsed() < ASK_GUARD && !is_stop {
                return;
            }
            let opts = a.options();
            match k.code {
                _ if is_stop => self.answer_ask(Answer::No),
                KeyCode::Up => a.sel = a.sel.saturating_sub(1),
                KeyCode::Down => a.sel = (a.sel + 1).min(opts.len() - 1),
                KeyCode::Enter => {
                    let r = opts[a.sel].0;
                    self.answer_ask(r);
                }
                KeyCode::Char(c @ '1'..='9') => {
                    if let Some(o) = opts.get(c as usize - '1' as usize) {
                        self.answer_ask(o.0);
                    }
                }
                _ => {}
            }
            return;
        }
        if !(ctrl && k.code == KeyCode::Char('c')) {
            self.exit_hint = None;
        }
        let popup = !self.popup.is_empty();
        match k.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy {
                    self.stop();
                } else if !self.ed.buf.is_empty() {
                    self.ed.clear();
                } else if self.exit_hint.is_some() {
                    self.quit = true;
                } else {
                    self.exit_hint = Some(Instant::now());
                }
            }
            KeyCode::Char('d') if ctrl => {
                if self.ed.buf.is_empty() {
                    self.quit = !self.busy;
                } else {
                    self.ed.delete();
                }
            }
            KeyCode::Esc if popup => {
                self.popup.clear();
                self.popup_off = true;
                return;
            }
            KeyCode::Esc => {
                if self.busy {
                    self.stop();
                }
            }
            KeyCode::Up if popup => {
                self.popup_sel = self.popup_sel.saturating_sub(1);
                return;
            }
            KeyCode::Down if popup => {
                self.popup_sel = (self.popup_sel + 1).min(self.popup.len() - 1);
                return;
            }
            KeyCode::Tab if popup => self.accept(),
            KeyCode::Enter if popup && !alt => {
                let (pick, whole) = (self.popup[self.popup_sel].0.clone(), self.ed.text());
                self.accept();
                // Perintah tanpa argumen langsung dijalankan.
                if pick.starts_with('/') && !matches!(pick.as_str(), "/model" | "/api") || whole == pick {
                    let s = self.ed.take();
                    self.submit(s);
                }
            }
            KeyCode::Enter if alt => self.ed.insert("\n"),
            KeyCode::Char('j') if ctrl => self.ed.insert("\n"),
            KeyCode::Enter => {
                if self.ed.cur > 0 && self.ed.buf[self.ed.cur - 1] == '\\' {
                    self.ed.backspace();
                    self.ed.insert("\n");
                } else {
                    let s = self.ed.take();
                    self.submit(s);
                }
            }
            KeyCode::Char('a') if ctrl => self.ed.home(),
            KeyCode::Char('e') if ctrl => self.ed.end(),
            KeyCode::Char('u') if ctrl => self.ed.kill(Editor::home),
            KeyCode::Char('k') if ctrl => self.ed.kill(Editor::end),
            KeyCode::Char('w') if ctrl => self.ed.kill(Editor::word_left),
            KeyCode::Backspace if alt => self.ed.kill(Editor::word_left),
            KeyCode::Backspace => self.ed.backspace(),
            KeyCode::Delete => self.ed.delete(),
            KeyCode::Left if ctrl || alt => self.ed.word_left(),
            KeyCode::Right if ctrl || alt => self.ed.word_right(),
            KeyCode::Left => self.ed.left(),
            KeyCode::Right => self.ed.right(),
            KeyCode::Home => self.ed.home(),
            KeyCode::End => self.ed.end(),
            KeyCode::Up => self.ed.up(),
            KeyCode::Down => self.ed.down(),
            KeyCode::Char(c) if !ctrl && !alt => self.ed.insert(c.encode_utf8(&mut [0; 4])),
            _ => return,
        }
        self.popup_off = false;
        self.refresh_popup();
    }

    fn accept(&mut self) {
        let pick = self.popup[self.popup_sel].0.clone();
        let tail = if pick.starts_with('@') || matches!(pick.as_str(), "/model" | "/api") { " " } else { "" };
        self.ed.replace_token(&format!("{pick}{tail}"));
        self.popup.clear();
    }

    fn refresh_popup(&mut self) {
        self.popup.clear();
        self.popup_sel = 0;
        if self.popup_off {
            return;
        }
        let (at, tok) = self.ed.token();
        if at == 0 && tok.starts_with('/') && self.ed.cur == self.ed.buf.len() {
            for (k, d) in COMMANDS {
                if k.starts_with(&tok) {
                    self.popup.push((k.into(), d.into()));
                }
            }
        } else if let Some(q) = tok.strip_prefix('@') {
            let cwd = self.cwd.clone();
            let q = q.to_lowercase();
            let files = self.files.get_or_insert_with(|| walk(&cwd));
            let mut hits: Vec<&String> = files.iter().filter(|f| f.to_lowercase().contains(&q)).collect();
            // Cocok di nama file didahulukan, lalu path terpendek.
            hits.sort_by_key(|f| {
                let name = f.rsplit('/').next().unwrap_or(f).to_lowercase();
                (!name.starts_with(&q), !name.contains(&q), f.len())
            });
            self.popup = hits.into_iter().take(30).map(|f| (format!("@{f}"), String::new())).collect();
        }
    }
}

/// Daftar file relatif, dibatasi agar tetap cepat di folder besar.
fn walk(root: &Path) -> Vec<String> {
    let mut v = vec![];
    let mut dirs = VecDeque::from([root.to_path_buf()]);
    let mut seen = 0;
    while let Some(d) = dirs.pop_front() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            seen += 1;
            if seen > 20_000 || v.len() >= 5_000 {
                return v;
            }
            let name = e.file_name().to_string_lossy().into_owned();
            // Nama dengan spasi atau karakter kontrol tak bisa ditulis sebagai @token.
            if name.contains(|c: char| c.is_whitespace() || c.is_control()) {
                continue;
            }
            let Ok(t) = e.file_type() else { continue };
            if t.is_dir() {
                if !name.starts_with('.') && !matches!(name.as_str(), "target" | "node_modules" | "__pycache__") {
                    dirs.push_back(e.path());
                }
            } else if let Ok(r) = e.path().strip_prefix(root) {
                let p = r.display().to_string();
                v.push(sanitize(&if cfg!(windows) { p.replace('\\', "/") } else { p }));
            }
        }
    }
    v
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { T[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(rows: &[Row]) -> Vec<String> {
        rows.iter().map(|r| {
            let mut s = r.s.clone();
            for c in [RST, BOLD, DIM, RED, GREEN, YELLOW, CYAN] {
                s = s.replace(c, "");
            }
            assert_eq!(sw(&s), r.w, "lebar salah: {s:?}");
            s
        }).collect()
    }

    #[test]
    fn bungkus() {
        let w = |t: &str, n| plain(&wrap(&[("", t.into())], n, ("", "● ".into()), "  "));
        assert_eq!(w("halo dunia ini", 12), ["● halo dunia", "  ini"]);
        assert_eq!(w("", 12), ["● "]);
        assert_eq!(w("abcdefghijkl", 8), ["● abcdef", "  ghijkl"]);
        for r in wrap(&[(BOLD, "日本語のテキスト é x".into())], 9, ("", "● ".into()), "  ") {
            assert!(r.w <= 9);
        }
    }

    #[test]
    fn kontrol_dibuang() {
        assert_eq!(sanitize("a\x1b[2Jb\r\x07\u{9b}c\u{202e}"), "a[2Jbc\u{fffd}");
    }

    #[test]
    fn markdown_tidak_kehilangan_teks() {
        for (s, want) in [("a **b** `c`", "a b c"), ("2 * 3 ** x", "2 * 3 ** x"), ("`tak tutup", "`tak tutup"), ("**", "**"), ("é*", "é*")] {
            let t: String = markdown(s, &mut false).into_iter().map(|x| x.1).collect();
            assert_eq!(t, want);
        }
    }

    #[test]
    fn b64() {
        assert_eq!([base64(b"M"), base64(b"Ma"), base64(b"Man"), base64(b"")], ["TQ==", "TWE=", "TWFu", ""]);
    }
}
