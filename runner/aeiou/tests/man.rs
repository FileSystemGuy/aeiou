//! The manual page of the runner (`man/aeiou.1.md`) against the binary's `--help`: every long
//! option of every subcommand is defined in the page's section for that subcommand (or in its
//! "Global options"), spelled as the help spells it (`--[no-]x` for a boolean), and the page
//! defines nothing the binary does not have; every `--flag` the page names anywhere, in prose
//! included, is an option of some subcommand; the page has the sections of a man page. The
//! Python suite (`builder/tests/test_man.py`) does the same for the Python tools' pages and
//! checks the flags named in every page of `man/` against the union.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

const SUBCOMMANDS: [&str; 4] = ["check", "dry-run", "datagen", "run"];

fn help(sub: &str) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_aeiou")).arg(sub).arg("--help").env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "aeiou {sub} --help failed: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

fn page() -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../man/aeiou.1.md")).unwrap()
}

/// The long option on a help line: clap puts an option at 2 or 6 columns of indentation,
/// `-x, --long` or `--long`, continuation lines far deeper. `help` and `version` are not
/// options of the tool.
fn help_flag(line: &str) -> Option<String> {
    let indent = line.len() - line.trim_start().len();
    if !(2..=6).contains(&indent) {
        return None;
    }
    let mut rest = line.trim_start();
    if rest.len() > 4 && rest.as_bytes()[0] == b'-' && rest.as_bytes()[1] != b'-' && &rest[2..4] == ", " {
        rest = &rest[4..];
    }
    let rest = rest.strip_prefix("--")?;
    let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '[' || c == ']')).unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty() || name == "help" || name == "version" {
        return None;
    }
    Some(name.to_string())
}

fn help_flags(sub: &str) -> BTreeSet<String> {
    help(sub).lines().filter_map(help_flag).collect()
}

/// The option a list item of the page defines: `- **--flag** ...` or `- **-x**, **--flag** ...`.
fn defined_flag(line: &str) -> Option<String> {
    let rest = line.strip_prefix("- **")?;
    let rest = match rest.strip_prefix("--") {
        Some(r) => r,
        None => rest.split_once("**, **--")?.1,
    };
    let (name, _) = rest.split_once("**")?;
    if name == "help" || name == "version" {
        return None;
    }
    Some(name.to_string())
}

/// The page split at its `### ` headings: (heading, body).
fn sections(page: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in page.lines() {
        if let Some(h) = line.strip_prefix("### ") {
            out.push((h.trim().to_string(), String::new()));
        } else if let Some(last) = out.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }
    out
}

fn defined_in(sections: &[(String, String)], heading: &str) -> BTreeSet<String> {
    let (_, body) = sections.iter().find(|(h, _)| h == heading).unwrap_or_else(|| panic!("no `### {heading}` section in man/aeiou.1.md"));
    body.lines().filter_map(defined_flag).collect()
}

/// Every `--name` token of the page, `[no-]` stripped.
fn mentioned(page: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = page.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if &bytes[i..i + 2] == b"--" && (i == 0 || bytes[i - 1] != b'-') {
            let mut j = i + 2;
            if page[j..].starts_with("[no-]") {
                j += 5;
            }
            let start = j;
            while j < bytes.len() && (bytes[j].is_ascii_lowercase() || bytes[j].is_ascii_digit() || bytes[j] == b'-') {
                j += 1;
            }
            let name = page[start..j].trim_end_matches('-');
            if !name.is_empty() && bytes[start].is_ascii_lowercase() {
                out.insert(name.to_string());
            }
            i = j.max(i + 2);
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn every_subcommands_options_are_the_pages() {
    let page = page();
    let sections = sections(&page);
    let global = defined_in(&sections, "Global options");
    assert_eq!(global, ["config"].into_iter().map(String::from).collect(), "the global options");
    let mut all_help = BTreeSet::new();
    for sub in SUBCOMMANDS {
        let in_help = help_flags(sub);
        assert!(in_help.contains("config"), "aeiou {sub} --help lacks --config");
        all_help.extend(in_help.iter().cloned());
        let in_page: BTreeSet<String> = defined_in(&sections, &format!("aeiou {sub}")).union(&global).cloned().collect();
        let missing: Vec<_> = in_help.difference(&in_page).collect();
        let stale: Vec<_> = in_page.difference(&in_help).collect();
        assert!(missing.is_empty() && stale.is_empty(), "aeiou {sub}: options in --help and not in man/aeiou.1.md: {missing:?}; in the page and not in --help: {stale:?}");
    }
    // the page names no option that no subcommand has (`--x` / `--no-x` is the notation's placeholder)
    let booleans: BTreeSet<String> = all_help.iter().filter_map(|f| f.strip_prefix("[no-]").map(String::from)).collect();
    let mut allowed: BTreeSet<String> = all_help.iter().map(|f| f.trim_start_matches("[no-]").to_string()).collect();
    allowed.extend(booleans.iter().map(|b| format!("no-{b}")));
    allowed.insert("x".into());
    allowed.insert("no-x".into());
    allowed.insert("help".into());
    allowed.insert("version".into());
    let unknown: Vec<_> = mentioned(&page).difference(&allowed).cloned().collect();
    assert!(unknown.is_empty(), "man/aeiou.1.md names options no subcommand has: {unknown:?}");
}

#[test]
fn the_page_has_the_sections_of_a_man_page() {
    let page = page();
    assert!(page.starts_with("# aeiou(1)\n"), "the title line");
    for h in ["NAME", "SYNOPSIS", "DESCRIPTION", "COMMANDS", "OPTIONS", "ENVIRONMENT", "FILES", "EXIT STATUS", "EXAMPLES", "SEE ALSO"] {
        assert!(page.contains(&format!("\n## {h}\n")), "no `## {h}` section");
    }
    assert!(page.contains("\naeiou - "), "the NAME line is `aeiou - description`");
    // every subcommand of the top-level help has a COMMANDS entry and an OPTIONS section
    let top = help("--help");
    for sub in SUBCOMMANDS {
        assert!(top.lines().any(|l| l.trim_start().starts_with(&format!("{sub} ")) || l.trim_start().starts_with(&format!("{sub}\t"))), "`aeiou --help` lists {sub}");
        assert!(page.contains(&format!("\n### {sub}\n")), "COMMANDS has `### {sub}`");
        assert!(page.contains(&format!("\n### aeiou {sub}\n")), "OPTIONS has `### aeiou {sub}`");
    }
}
