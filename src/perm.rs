//! Kebijakan izin. Ini bukan sandbox: perintah bash yang disetujui berjalan
//! dengan hak akses pengguna.

use crate::tools::{Action, Prepared};
use std::collections::HashSet;

#[derive(Default)]
pub struct Perm {
    pub allow_edit: bool,
    pub allow_bash: bool,
    pub allow_memory: bool,
    bash_exact: HashSet<String>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Answer {
    Yes,
    Always,
    No,
}

impl Perm {
    pub fn auto(&self, p: &Prepared) -> bool {
        match &p.action {
            Action::Read { .. } | Action::Grep { .. } => !p.guarded,
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
            Action::Read { .. } | Action::Grep { .. } => None,
            Action::Write { .. } => Some("Ya, selalu untuk edit di folder ini selama sesi"),
            Action::Bash { .. } => Some("Ya, selalu untuk perintah persis ini selama sesi"),
        }
    }

    pub fn remember(&mut self, p: &Prepared) {
        match &p.action {
            _ if p.memory => self.allow_memory = true,
            _ if p.guarded => {}
            Action::Read { .. } | Action::Grep { .. } => {}
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
}
