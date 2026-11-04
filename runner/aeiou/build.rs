//! The application records (`applications/` at the repository's root, `DESIGN_REVIEW.md` §3.88)
//! built into the crate, so that `aeiou-trace digest` carries them wherever its binary goes
//! (§3.89 stage 3): `attributes.toml` and every file of every application's directory, by its
//! path under `applications/`, as `src/applications.rs` reads a directory. And the capture's
//! BPF object (`../capture-ebpf/`, DESIGN_REVIEW.md §3.96 "Phase 3"), built for the BPF target
//! with the nightly that crate's `rust-toolchain.toml` names and `bpf-linker` from the `PATH`
//! (`dev.sh` puts the pinned one there), and embedded by `src/ebpf.rs`.
use std::path::Path;

fn main() {
    ebpf();
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

/// `cargo build` of the probe crate in its own directory, so rustup takes its toolchain file and
/// cargo its own workspace and lock, with only the environment a build needs: nothing of this
/// build's (its target, its `RUSTFLAGS`, coverage's instrumentation) reaches the BPF target.
fn ebpf() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../capture-ebpf").canonicalize().expect("runner/capture-ebpf");
    for f in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src"] {
        println!("cargo:rerun-if-changed={}", dir.join(f).display());
    }
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    assert!(arch == "x86_64" || arch == "aarch64", "the capture's BPF object reads system call arguments on x86_64 and aarch64 only, not {arch}");
    let target = out.join("capture-ebpf");
    let mut cmd = std::process::Command::new("cargo");
    cmd.current_dir(&dir).env_clear();
    for k in ["PATH", "HOME", "RUSTUP_HOME", "CARGO_HOME", "TMPDIR"] {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    let flags = [format!("--cfg=bpf_target_arch=\"{arch}\""), "-Cdebuginfo=2".into(), "-Clink-arg=--btf".into()].join("\x1f");
    cmd.env("CARGO_ENCODED_RUSTFLAGS", flags);
    cmd.args(["build", "--release", "--locked", "--target", "bpfel-unknown-none", "-Z", "build-std=core", "--target-dir"]).arg(&target);
    let st = cmd.status().expect("cargo, for the capture's BPF object");
    assert!(st.success(), "the capture's BPF object did not build (runner/capture-ebpf; it needs the nightly its rust-toolchain.toml names and bpf-linker on the PATH: ./dev.sh setup)");
    std::fs::copy(target.join("bpfel-unknown-none/release/aeiou-capture"), out.join("aeiou-capture.bpf.o")).unwrap();
}
