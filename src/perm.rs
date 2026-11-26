//! Kebijakan izin. Ini bukan sandbox: perintah bash yang disetujui berjalan
//! dengan hak akses pengguna.

use crate::tools::{Action, Prepared};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub struct Perm {
    pub allow_edit: bool,
    pub allow_bash: bool,
    pub allow_memory: bool,
    pub allow_web: bool,
    /// Mode auto: semua disetujui kecuali path yang dijaga. Dibagi dengan UI agar
    /// Shift+Tab berlaku seketika, juga di tengah giliran.
    pub all: Arc<AtomicBool>,
    bash_exact: HashSet<String>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Answer {
    Yes,
    Always,
    No,
}

impl Perm {
    pub fn is_all(&self) -> bool {
        self.all.load(Ordering::Relaxed)
    }

    pub fn auto(&self, p: &Prepared) -> bool {
        match &p.action {
            // Di luar folder kerja, .git, atau file rahasia: tetap ditanya, juga di mode auto.
            _ if self.is_all() && (p.memory || !p.guarded) => true,
            Action::Read { .. } | Action::Grep { .. } | Action::Essence { .. } => !p.guarded,
            Action::Web { .. } => self.allow_web,
            // Memori untuk semua proyek tidak ikut izin edit folder ini.
            Action::Write { .. } if p.memory && self.allow_memory => true,
            Action::Write { .. } => self.allow_edit && !p.guarded,
            Action::Bash { cmd, .. } => self.allow_bash || self.bash_exact.contains(cmd),
        }
    }

    /// Teks opsi "selalu", atau None bila tidak ditawarkan.
    pub fn always_label(p: &Prepared) -> Option<&'static str> {
        match &p.action {
            _ if p.memory => Some("Ya, selalu simpan memori selama sesi"),
            _ if p.guarded => None,
            Action::Read { .. } | Action::Grep { .. } | Action::Essence { .. } => None,
            Action::Web { .. } => Some("Ya, selalu untuk akses internet selama sesi"),
            Action::Write { .. } => Some("Ya, selalu untuk edit di folder ini selama sesi"),
            Action::Bash { .. } => Some("Ya, selalu untuk perintah persis ini selama sesi"),
        }
    }

    pub fn remember(&mut self, p: &Prepared) {
        match &p.action {
            _ if p.memory => self.allow_memory = true,
            _ if p.guarded => {}
            Action::Read { .. } | Action::Grep { .. } | Action::Essence { .. } => {}
            Action::Web { .. } => self.allow_web = true,
            Action::Write { .. } => self.allow_edit = true,
            Action::Bash { cmd, .. } => {
                self.bash_exact.insert(cmd.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bash(c: &str) -> Prepared {
        Prepared { action: Action::Bash { cmd: c.into(), timeout: 1 }, title: "".into(), guarded: false, memory: false }
    }
    fn write(guarded: bool) -> Prepared {
        Prepared {
            action: Action::Write { path: "x".into(), old: None, new: "".into(), is_edit: false },
            title: "".into(),
            guarded,
            memory: false,
        }
    }

    #[test]
    fn selalu_dibatasi() {
        let mut p = Perm::default();
        assert!(!p.auto(&bash("ls")));
        p.remember(&bash("ls"));
        assert!(p.auto(&bash("ls")) && !p.auto(&bash("ls; rm -rf x")));
        p.remember(&write(true));
        assert!(!p.auto(&write(false)));
        let mut m = write(true);
        m.memory = true;
        assert!(!p.auto(&m));
        p.remember(&m);
        assert!(p.auto(&m) && !p.auto(&write(false)) && !p.auto(&write(true)));
        p.remember(&write(false));
        assert!(p.auto(&write(false)) && !p.auto(&write(true)));
    }

    #[test]
    fn mode_auto_tidak_meloloskan_path_dijaga() {
        let p = Perm::default();
        let web = Prepared { action: Action::Web { tool: "curl".into(), args: serde_json::json!({}), lines: vec![] }, title: "".into(), guarded: false, memory: false };
        assert!(!p.auto(&bash("make")) && !p.auto(&write(false)) && !p.auto(&web));
        p.all.store(true, Ordering::Relaxed);
        assert!(p.auto(&bash("make")) && p.auto(&write(false)) && p.auto(&web));
        assert!(!p.auto(&write(true)));
        let mut m = write(true);
        m.memory = true;
        assert!(p.auto(&m));
        p.all.store(false, Ordering::Relaxed);
        assert!(!p.auto(&bash("make")));
    }
}
