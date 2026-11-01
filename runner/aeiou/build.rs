//! The application records (`applications/` at the repository's root, `DESIGN_REVIEW.md` §3.88)
//! built into the crate, so that `aeiou-trace digest` carries them wherever its binary goes
//! (§3.89 stage 3): `attributes.toml` and every file of every application's directory, by its
//! path under `applications/`, as `src/applications.rs` reads a directory.
use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../applications");
    println!("cargo:rerun-if-changed={}", root.display());
    assert!(root.is_dir(), "{}: the application records are built into the crate, and the checkout has none", root.display());
    let mut files = vec!["attributes.toml".to_string()];
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    for d in dirs {
        println!("cargo:rerun-if-changed={}", d.display());
        let mut names: Vec<_> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_file()).collect();
        names.sort();
        for f in names {
            println!("cargo:rerun-if-changed={}", f.display());
            files.push(f.strip_prefix(&root).unwrap().to_str().unwrap().to_string());
        }
    }
    let mut out = String::from("pub static SHIPPED: &[(&str, &str)] = &[\n");
    for f in &files {
        out += &format!("    ({f:?}, include_str!({:?})),\n", root.join(f).canonicalize().unwrap());
    }
    out += "];\n";
    std::fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("applications.rs"), out).unwrap();
}
