#!/bin/sh
# dev.sh: the checkout's developer script (DESIGN_REVIEW.md §3.83). A clone needs rustup and uv;
# everything else is a version the repository names, which they and this script fetch:
# Rust from rust-toolchain.toml, the crates from runner/Cargo.lock, uv's own version, the Python,
# and the packages from builder/ (pyproject.toml, .python-version, uv.lock), and the downloads
# pinned below by URL and SHA-256. To run the benchmark, use a release's tarball instead
# (QUICKSTART.md): it needs none of this.
#
#   dev.sh setup                build the runner and the venv, aeiou-header beside the binary
#   dev.sh check [JOB]...       CI's jobs, the same steps (each CI job is `./dev.sh check JOB`)
#   dev.sh cov                  coverage of the Rust and the Python, per file
#   dev.sh fetch TOOL           one pinned download, checked; prints its path
#
# Usage errors are printed in the suite's frame (runner/REFERENCE.md §15), exit status 2; a
# missing prerequisite or a failed step is 1.
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd -P)
CACHE=${AEIOU_DEV_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/aeiou-dev}
PY=$ROOT/builder/.venv/bin
BIN=$ROOT/runner/target/release/aeiou
USAGE="Usage: dev.sh <COMMAND> [ARGS]..."
JOBS="abstracts runner object integration all"
TOOLS="minio nextest llvm-cov"

usage_error() {
    printf '%s\n\n%s\n\nFor more information, try '"'"'dev.sh --help'"'"'.\n' "dev.sh: $1" "$USAGE" >&2
    exit 2
}
die() {
    printf 'dev.sh: %s\n' "$1" >&2
    exit 1
}
commas() { echo "$1" | sed 's/ /, /g'; }

# The downloads: the tool, the architecture (uname -m), the URL of one release, and the SHA-256 of
# the file it serves (MinIO's last release with binaries: the project archived its repository).
ARCHES="x86_64 aarch64"
PINS=
pin() {
    PINS="$PINS$1 $2 $3 $4
"
}
pin minio x86_64 https://github.com/minio/minio/releases/download/RELEASE.2025-09-07T16-13-09Z/minio.linux-amd64.RELEASE.2025-09-07T16-13-09Z 7c5bd8512c6e966455b1d198209358b2d191c77a83ab377c4073281065fb855f
pin minio aarch64 https://github.com/minio/minio/releases/download/RELEASE.2025-09-07T16-13-09Z/minio.linux-arm64.RELEASE.2025-09-07T16-13-09Z 5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d
pin nextest x86_64 https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-0.9.149/cargo-nextest-0.9.149-x86_64-unknown-linux-gnu.tar.gz 32d1d8c0567b7a80bee2bad8bb09125c846da96bc009b2de11ec08557290c770
pin nextest aarch64 https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-0.9.149/cargo-nextest-0.9.149-aarch64-unknown-linux-gnu.tar.gz 6f82e375187cf3a1f3e336879a51940c315a5d0ce7253f7e517ad28295b74189
pin llvm-cov x86_64 https://github.com/taiki-e/cargo-llvm-cov/releases/download/v0.9.1/cargo-llvm-cov-x86_64-unknown-linux-gnu.tar.gz b3f68e625481fed9b16444174f3fa5ebcdbde4a1878803a35eabe2dcefcdc41a
pin llvm-cov aarch64 https://github.com/taiki-e/cargo-llvm-cov/releases/download/v0.9.1/cargo-llvm-cov-aarch64-unknown-linux-gnu.tar.gz abf5f13c1520f8756d2192bfaaeadb0208f5b7eaa8cadc15bf847befa42b7360

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<EOT
Build and check a checkout of aeiou at the versions it pins, as CI does

$USAGE

Commands:
  setup           Build the runner (cargo --locked) and builder/.venv (uv sync --locked, every extra and the
                  coverage group), and put aeiou-header beside the binary
  check [JOB]...  Run CI's jobs, the same steps: $(commas "$JOBS") (the default)
  cov             Coverage of the Rust (cargo-llvm-cov) and the Python (pytest-cov), per file, nothing skipped
  fetch <TOOL>    Download one pinned tool, $(commas "$TOOLS"), for this architecture, checked by its SHA-256,
                  and print its path

Options:
  -h, --help  Print help

Needs rustup (https://rustup.rs) and uv (https://docs.astral.sh/uv/getting-started/installation/), and strace for
the integration job and cov; the versions are the checkout's. Downloads, x86_64 or aarch64, are kept in
\$AEIOU_DEV_CACHE (default \${XDG_CACHE_HOME:-~/.cache}/aeiou-dev), a directory per architecture, and checked again
each time they are used. To run the benchmark, a release's tarball needs none of this (QUICKSTART.md).
EOT
    exit 0
fi

# every prerequisite the command needs that PATH lacks, listed together, before anything runs
need() {
    missing=
    for t in "$@"; do
        command -v "$t" >/dev/null 2>&1 && continue
        case $t in
        cargo) missing="$missing
  cargo   Rust, through rustup: https://rustup.rs (it installs the version rust-toolchain.toml names)" ;;
        uv) missing="$missing
  uv      https://docs.astral.sh/uv/getting-started/installation/ (builder/pyproject.toml names its version)" ;;
        strace) missing="$missing
  strace  your distribution's package (the traces' tests run it)" ;;
        esac
    done
    [ -z "$missing" ] || die "not on PATH, and needed:$missing"
}

# fetch TOOL: this architecture's pinned file, downloaded once into $CACHE/ARCH (a cache on a
# home directory hosts of both kinds mount) and checked against its hash every time; a tarball's
# one executable is unpacked beside it. Prints the executable's path.
fetch() {
    arch=$(uname -m)
    line=$(printf '%s' "$PINS" | grep "^$1 $arch ") ||
        die "$1: no download is pinned for $arch, only $(echo "$ARCHES" | sed 's/ / and /')"
    url=$(echo "$line" | cut -d' ' -f3)
    sha=$(echo "$line" | cut -d' ' -f4)
    dir=$CACHE/$arch
    mkdir -p "$dir"
    case $url in
    *.tar.gz) file=$dir/$1.tar.gz ;;
    *) file=$dir/$1 ;;
    esac
    if ! { [ -f "$file" ] && printf '%s  %s\n' "$sha" "$file" | sha256sum -c --status 2>/dev/null; }; then
        rm -f "$file"
        part=$dir/.download.$$
        if ! curl -sSfL --retry 3 -o "$part" "$url" 2>"$part.err"; then
            err=$(cat "$part.err")
            rm -f "$part" "$part.err"
            die "$1: the download failed ($url): $err"
        fi
        rm -f "$part.err"
        if ! printf '%s  %s\n' "$sha" "$part" | sha256sum -c --status 2>/dev/null; then
            rm -f "$part"
            die "$1: the download's SHA-256 is not the pin's ($url)"
        fi
        mv "$part" "$file"
    fi
    case $file in
    *.tar.gz)
        exe=$(tar tzf "$file" | head -n 1)
        tar xzf "$file" -C "$dir" "$exe"
        echo "$dir/$exe" ;;
    *)
        chmod +x "$file"
        echo "$file" ;;
    esac
}

py_sync() { (cd "$ROOT/builder" && uv sync --locked --all-extras --group coverage); }
rust_build() { (cd "$ROOT/runner" && cargo build --release --locked --bin aeiou); }
link_header() { ln -sf ../../../builder/.venv/bin/aeiou-header "$ROOT/runner/target/release/aeiou-header"; }

# The jobs of .github/workflows/ci.yml, one function each.
job_abstracts() {
    py_sync
    "$PY/python" "$ROOT/schema/check.py"
    (cd "$ROOT/builder" && AEIOU_REQUIRE=py-formats .venv/bin/pytest -q -n auto)
    (cd "$ROOT/builder" && .venv/bin/aeiou-build --hermetic --twice --check $(find ../workloads -name '*.py' -not -path '*/kit/*' | LC_ALL=C sort))
    # the parameter files fit their abstracts
    (cd "$ROOT/builder" && for f in $(find ../workloads -name '*params.json' | LC_ALL=C sort); do
        a=$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["abstract"])' "$f")
        .venv/bin/aeiou-params check "$(find ../workloads -name "$a.ast.json")" "$f" || exit 1
    done)
}
job_runner() {
    minio=$(fetch minio) || exit 1
    nextest=$(fetch nextest) || exit 1
    # the whole suite, the object engine's tests against MinIO included
    (cd "$ROOT/runner" && TEST_MINIO_BIN=$minio "$nextest" nextest run --release --locked)
    # a build without the object engine refuses an object endpoint
    (cd "$ROOT/runner" && cargo test --release --locked --no-default-features --lib endpoint::)
}
job_object() {
    minio=$(fetch minio) || exit 1
    nextest=$(fetch nextest) || exit 1
    # the object engine without `formats`, so that build is compiled too
    (cd "$ROOT/runner" && TEST_MINIO_BIN=$minio "$nextest" nextest run --release --locked --no-default-features --features object -E 'test(/object/) | test(/endpoint/)')
    # the Python writer into a bucket, read back by the engine to the dry-run fingerprint
    py_sync
    rust_build
    link_header
    (cd "$ROOT/builder" && AEIOU_REQUIRE=runner,object,formats,header,minio,py-formats,py-s3 TEST_MINIO_BIN=$minio AEIOU_RUNNER="$BIN" .venv/bin/pytest -q -rs tests/test_object.py)
}
job_integration() {
    rust_build
    py_sync
    link_header
    # the Rust checker agrees with the reference checker on every committed AST
    out=$(mktemp -d)
    "$PY/python" "$ROOT/schema/check.py" > "$out/py"
    (cd "$ROOT" && "$BIN" check $(find workloads -name '*.ast.json' | LC_ALL=C sort)) > "$out/rs.all"
    sed '1,/^$/d' "$out/rs.all" > "$out/rs"                          # past the options block
    diff "$out/py" "$out/rs" || { rm -rf "$out"; die "the Rust checker and schema/check.py disagree"; }
    rm -rf "$out"
    # every Python test against the binary, none skipped for what this job built
    (cd "$ROOT/builder" && AEIOU_REQUIRE=runner,formats,strace,header,py-formats AEIOU_RUNNER="$BIN" .venv/bin/pytest -q -n auto -rs)
    # the workload catalog, every shape re-measured at its check length
    (cd "$ROOT/builder" && .venv/bin/aeiou-catalog check --workloads ../workloads --runner "$BIN")
}

cov() {
    minio=$(fetch minio) || exit 1
    llvm=$(fetch llvm-cov) || exit 1
    (cd "$ROOT/runner" && TEST_MINIO_BIN=$minio "$llvm" llvm-cov --release --locked --workspace --all-targets --summary-only \
        --no-default-ignore-filename-regex --ignore-filename-regex '/\.cargo/|/rustc/|/\.rustup/')
    rust_build
    py_sync
    link_header
    out=$(mktemp -d)
    # the CLI subprocesses measured too (coverage's patch = subprocess), the scripts and kits under workloads/
    cat > "$out/rc" <<EOF
[run]
parallel = true
patch = subprocess
data_file = $out/.coverage
source =
    $ROOT/builder/aeiou
    $ROOT/builder/tests
    $ROOT/workloads
EOF
    (cd "$ROOT/builder" && AEIOU_REQUIRE=runner,formats,object,strace,header,minio,py-formats,py-s3 TEST_MINIO_BIN=$minio \
        AEIOU_RUNNER="$BIN" .venv/bin/pytest -q -n auto -rs -p no:cacheprovider --cov --cov-config="$out/rc" --cov-report=) \
        || { rm -rf "$out"; exit 1; }
    "$PY/coverage" combine -q --rcfile="$out/rc" 2>/dev/null || true
    "$PY/coverage" report --rcfile="$out/rc"
    rm -rf "$out"
}

[ $# -gt 0 ] || usage_error "the following required arguments were not provided:
  <COMMAND>  one of setup, check, cov, fetch"
cmd=$1
shift
case $cmd in
setup | cov)
    [ $# -eq 0 ] || usage_error "unexpected argument '$1' found"
    need cargo uv
    if [ "$cmd" = setup ]; then
        rust_build
        py_sync
        link_header
    else
        need strace
        cov
    fi ;;
check)
    [ $# -gt 0 ] || set -- all
    for j in "$@"; do
        case " $JOBS " in
        *" $j "*) ;;
        *) usage_error "invalid value '$j' for '[JOB]...'
  [possible values: $(commas "$JOBS")]" ;;
        esac
    done
    # what the jobs asked for need, all of it named at once: abstracts no Rust, runner no
    # Python (CI sets up neither where its job has none)
    tools=
    for j in "$@"; do
        case $j in
        abstracts) want=uv ;;
        runner) want=cargo ;;
        object) want="cargo uv" ;;
        integration | all) want="cargo uv strace" ;;
        esac
        for t in $want; do
            case " $tools " in *" $t "*) ;; *) tools="$tools $t" ;; esac
        done
    done
    need $tools
    for j in "$@"; do
        if [ "$j" = all ]; then
            set -- abstracts runner object integration
            for k in "$@"; do "job_$k"; done
        else
            "job_$j"
        fi
    done ;;
fetch)
    [ $# -gt 0 ] || usage_error "the following required arguments were not provided:
  <TOOL>  one of $(commas "$TOOLS")"
    case " $TOOLS " in
    *" $1 "*) ;;
    *) usage_error "invalid value '$1' for '<TOOL>'
  [possible values: $(commas "$TOOLS")]" ;;
    esac
    [ $# -eq 1 ] || usage_error "unexpected argument '$2' found"
    fetch "$1" ;;
-*) usage_error "unexpected argument '$cmd' found" ;;
*) usage_error "unrecognized subcommand '$cmd'" ;;
esac
