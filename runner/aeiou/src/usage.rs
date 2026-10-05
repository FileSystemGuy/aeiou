//! Usage errors: the command line is wrong, and nothing has run (`runner/REFERENCE.md` §15,
//! `DESIGN_REVIEW.md` §3.61). Every tool of the suite, Rust or Python, prints them in one
//! frame, so a new user meets the same message whichever tool spoke:
//!
//! ```text
//! aeiou datagen: the following required arguments were not provided:
//!   <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)
//!   --root <DIR>     directory the abstract's paths are relative to
//!                    (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)
//!
//! Usage: aeiou datagen [OPTIONS] --root <DIR> <ABSTRACT_PATH>
//!
//! For more information, try 'aeiou datagen --help'.
//! ```
//!
//! The first line names the command as typed, then the message; the usage line; the pointer
//! to the help. The exit status is 2 (the parsers' own convention), a failure during the
//! work is 1. A missing requirement is never reported alone: the command collects every
//! argument it still lacks, from any layer, and lists them together (`Missing`), so the next
//! attempt is the last one. The Python mirror is `builder/aeiou/usage.py`.

use std::fmt;

/// The command line is wrong. `Display` is the message alone; `render` puts it in the frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UsageError {}

/// `return Err(UsageError(format!(...)).into())`: `bail!` for a usage error.
#[macro_export]
macro_rules! usage {
    ($($arg:tt)*) => {
        return Err(::anyhow::Error::new($crate::usage::UsageError(format!($($arg)*))))
    };
}

/// `anyhow!` for a usage error: the error value, for `ok_or_else` and `map_err`.
pub fn err(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(UsageError(msg.into()))
}

/// The first line of the list of missing requirements (the parsers' wording).
pub const MISSING: &str = "the following required arguments were not provided:";

/// The frame: `prog` is the command as typed (`aeiou datagen`), `usage` its usage text
/// beginning `Usage: ` (several lines allowed), without a trailing newline.
pub fn render(prog: &str, message: &str, usage: &str) -> String {
    format!("{prog}: {message}\n\n{}\n\nFor more information, try '{prog} --help'.\n", usage.trim_end())
}

/// What the tool's own parser printed, reframed: clap's `error: ` becomes the command's name
/// and its `--help` pointer names the command, so a wrong flag reads like a missing one; the
/// usage line is put in when clap left it out (it does for an invalid value), since the frame
/// has it always.
pub fn reframe(rendered: &str, prog: &str, usage: &str) -> String {
    let body = match rendered.strip_prefix("error: ") {
        Some(b) => format!("{prog}: {b}"),
        None => rendered.to_string(),
    };
    let body = body.replace("try '--help'", &format!("try '{prog} --help'"));
    if body.contains("\nUsage: ") {
        return body;
    }
    match body.rsplit_once("\n\nFor more information") {
        Some((head, tail)) => format!("{head}\n\n{}\n\nFor more information{tail}", usage.trim_end()),
        None => body,
    }
}

/// One requirement the command line (or a lower layer) did not meet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// As the usage line shows it: `<ABSTRACT_PATH>`, `--root <DIR>`.
    pub shown: String,
    /// What it is, one line, no capital, no period.
    pub what: String,
    /// Where else it may come from, or why it is required now; shown in parentheses on the
    /// next line.
    pub also: Option<String>,
}

/// The requirements a command still lacks, collected before anything is done so that one
/// message lists them all.
#[derive(Debug, Default)]
pub struct Missing {
    items: Vec<Item>,
}

impl Missing {
    pub fn new() -> Missing {
        Missing::default()
    }

    /// A value that must be present: records it when `value` is `None`, and hands it back.
    pub fn want<T>(&mut self, value: Option<T>, shown: &str, what: &str, also: Option<&str>) -> Option<T> {
        if value.is_none() {
            self.need(false, shown, what, also);
        }
        value
    }

    /// A requirement stated as a condition: records it when `present` is false.
    pub fn need(&mut self, present: bool, shown: &str, what: &str, also: Option<&str>) {
        if !present {
            self.items.push(Item { shown: shown.into(), what: what.into(), also: also.map(str::to_string) });
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The message: the header, one line per item, the name column as wide as the longest.
    pub fn message(&self) -> String {
        let width = self.items.iter().map(|i| i.shown.len()).max().unwrap_or(0);
        let mut s = MISSING.to_string();
        for i in &self.items {
            s.push_str(&format!("\n  {:<width$}  {}", i.shown, i.what));
            if let Some(also) = &i.also {
                s.push_str(&format!("\n  {:<width$}  ({also})", ""));
            }
        }
        s
    }

    /// `Err(UsageError)` listing every item, or `Ok` when nothing is missing.
    pub fn check(&self) -> anyhow::Result<()> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(err(self.message()))
        }
    }
}

/// The abstract every subcommand but `check` takes, as `Missing` lists it.
pub const ABSTRACT: (&str, &str) = ("<ABSTRACT_PATH>", "the abstract (`.ast.json`, written by aeiou-build from a builder script)");
/// `--gpus`, as `run` and `dry-run` list it.
pub const GPUS: (&str, &str) = ("--gpus <GPUS>", "number of instances of every actor template whose count is `gpus`");

/// `--root`, as `run` and `datagen` list it: the parenthetical names the layers it may come from.
pub fn root(sub: &str) -> (&'static str, &'static str, String) {
    ("--root <DIR>", "directory the abstract's paths are relative to", format!("also $AEIOU_ROOT, or `root` in the [{sub}] table of the config file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_aligns_and_the_frame_names_the_command() {
        let mut m = Missing::new();
        let a: Option<i32> = m.want(None, ABSTRACT.0, ABSTRACT.1, None);
        assert!(a.is_none());
        let r = root("datagen");
        m.want::<i32>(None, r.0, r.1, Some(&r.2));
        assert_eq!(
            m.message(),
            "the following required arguments were not provided:\n  <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)\n  --root <DIR>     directory the abstract's paths are relative to\n                   (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)"
        );
        let text = render("aeiou datagen", &m.message(), "Usage: aeiou datagen [OPTIONS] --root <DIR> <ABSTRACT_PATH>\n");
        assert!(text.starts_with("aeiou datagen: the following"));
        assert!(text.ends_with("<ABSTRACT_PATH>\n\nFor more information, try 'aeiou datagen --help'.\n"));
        let e: anyhow::Error = m.check().unwrap_err();
        assert!(e.downcast_ref::<UsageError>().is_some());
        assert!(Missing::new().check().is_ok());
    }

    #[test]
    fn clap_output_is_reframed() {
        let usage = "Usage: aeiou run [OPTIONS]";
        let clap = "error: unexpected argument '--foo' found\n\nUsage: aeiou run [OPTIONS]\n\nFor more information, try '--help'.\n";
        assert_eq!(reframe(clap, "aeiou run", usage), "aeiou run: unexpected argument '--foo' found\n\nUsage: aeiou run [OPTIONS]\n\nFor more information, try 'aeiou run --help'.\n");
        // an invalid value: clap leaves the usage out, the frame puts it in
        let clap = "error: invalid value 'x' for '--gpus <GPUS>': invalid digit found in string\n\nFor more information, try '--help'.\n";
        assert_eq!(reframe(clap, "aeiou run", usage), "aeiou run: invalid value 'x' for '--gpus <GPUS>': invalid digit found in string\n\nUsage: aeiou run [OPTIONS]\n\nFor more information, try 'aeiou run --help'.\n");
    }
}
