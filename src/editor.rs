//! Penyunting teks input: multi-baris, riwayat, tanpa I/O.

#[derive(Default)]
pub struct Editor {
    pub buf: Vec<char>,
    pub cur: usize,
    hist: Vec<String>,
    hpos: Option<usize>,
    draft: String,
}

impl Editor {
    pub fn text(&self) -> String {
        self.buf.iter().collect()
    }

    pub fn set(&mut self, s: &str) {
        self.buf = s.chars().collect();
        self.cur = self.buf.len();
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.cur = 0;
        self.hpos = None;
    }

    /// Ambil isi dan catat ke riwayat.
    pub fn take(&mut self) -> String {
        let s = self.text();
        if !s.trim().is_empty() && self.hist.last() != Some(&s) {
            self.hist.push(s.clone());
        }
        self.clear();
        s
    }

    pub fn insert(&mut self, s: &str) {
        let s = s.replace("\r\n", "\n").replace('\r', "\n").replace('\t', "    ");
        for c in s.chars().filter(|c| *c == '\n' || !c.is_control()) {
            self.buf.insert(self.cur, c);
            self.cur += 1;
        }
    }

    pub fn backspace(&mut self) {
        if self.cur > 0 {
            self.cur -= 1;
            self.buf.remove(self.cur);
        }
    }

    pub fn delete(&mut self) {
        if self.cur < self.buf.len() {
            self.buf.remove(self.cur);
        }
    }

    pub fn left(&mut self) {
        self.cur = self.cur.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cur = (self.cur + 1).min(self.buf.len());
    }

    pub fn home(&mut self) {
        while self.cur > 0 && self.buf[self.cur - 1] != '\n' {
            self.cur -= 1;
        }
    }

    pub fn end(&mut self) {
        while self.cur < self.buf.len() && self.buf[self.cur] != '\n' {
            self.cur += 1;
        }
    }

    pub fn word_left(&mut self) {
        while self.cur > 0 && self.buf[self.cur - 1].is_whitespace() {
            self.cur -= 1;
        }
        while self.cur > 0 && !self.buf[self.cur - 1].is_whitespace() {
            self.cur -= 1;
        }
    }

    pub fn word_right(&mut self) {
        while self.cur < self.buf.len() && self.buf[self.cur].is_whitespace() {
            self.cur += 1;
        }
        while self.cur < self.buf.len() && !self.buf[self.cur].is_whitespace() {
            self.cur += 1;
        }
    }

    pub fn kill(&mut self, f: fn(&mut Self)) {
        let a = self.cur;
        f(self);
        let (lo, hi) = (a.min(self.cur), a.max(self.cur));
        self.buf.drain(lo..hi);
        self.cur = lo;
    }

    /// Naik/turun baris; di tepi, pindah riwayat.
    pub fn up(&mut self) {
        let col = self.col();
        self.home();
        if self.cur > 0 {
            self.cur -= 1;
            self.home();
            self.seek_col(col);
        } else if !self.hist.is_empty() {
            let i = match self.hpos {
                None => {
                    self.draft = self.text();
                    self.hist.len() - 1
                }
                Some(i) => i.saturating_sub(1),
            };
            self.hpos = Some(i);
            let s = self.hist[i].clone();
            self.set(&s);
        }
    }

    pub fn down(&mut self) {
        let col = self.col();
        self.end();
        if self.cur < self.buf.len() {
            self.cur += 1;
            self.seek_col(col);
        } else if let Some(i) = self.hpos {
            let s = if i + 1 < self.hist.len() {
                self.hpos = Some(i + 1);
                self.hist[i + 1].clone()
            } else {
                self.hpos = None;
                std::mem::take(&mut self.draft)
            };
            self.set(&s);
        }
    }

    fn col(&self) -> usize {
        self.buf[..self.cur].iter().rev().take_while(|c| **c != '\n').count()
    }

    fn seek_col(&mut self, n: usize) {
        for _ in 0..n {
            if self.cur < self.buf.len() && self.buf[self.cur] != '\n' {
                self.cur += 1;
            }
        }
    }

    /// Kata yang sedang diketik (dari spasi terakhir sampai kursor) dan awalnya.
    pub fn token(&self) -> (usize, String) {
        let s = self.buf[..self.cur].iter().rposition(|c| c.is_whitespace()).map_or(0, |i| i + 1);
        (s, self.buf[s..self.cur].iter().collect())
    }

    pub fn replace_token(&mut self, s: &str) {
        let (at, _) = self.token();
        self.buf.drain(at..self.cur);
        self.cur = at;
        self.insert(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sunting_dasar() {
        let mut e = Editor::default();
        e.insert("halo dunia\nbaris é");
        e.kill(Editor::word_left);
        assert_eq!(e.text(), "halo dunia\nbaris ");
        e.up();
        assert_eq!(e.cur, 6);
        e.kill(Editor::end);
        assert_eq!(e.text(), "halo d\nbaris ");
        e.insert("\x1b[31m\r\nx");
        assert_eq!(e.text(), "halo d[31m\nx\nbaris ");
    }

    #[test]
    fn riwayat_dan_token() {
        let mut e = Editor::default();
        e.insert("satu");
        e.take();
        e.insert("dua");
        e.take();
        e.insert("draf");
        e.up();
        e.up();
        assert_eq!(e.text(), "satu");
        e.down();
        e.down();
        assert_eq!(e.text(), "draf");
        e.insert(" @sr");
        assert_eq!(e.token().1, "@sr");
        e.replace_token("@src/a.rs ");
        assert_eq!(e.text(), "draf @src/a.rs ");
    }
}
