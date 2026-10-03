//! Site annotation. A draw's site is its structural path in the tree, the JSON pointer of the
//! object that holds the `draw`, `pick`, or `choose` key (`schema/README.md` §1, "No site
//! ids"). This walk mirrors the JSON structure exactly and stores `rng::site_hash(pointer)`
//! in each node's `site` cell.

use crate::ast::*;
use crate::rng::site_hash;

pub fn annotate(ast: &Ast) {
    let mut p = Path::default();
    for (name, param) in &ast.params {
        p.with(&["params", name, "default"], |p| pvalue(&param.default, p));
    }
    for (name, ds) in &ast.datasets {
        p.with(&["datasets", name], |p| match ds {
            Dataset::Files(f) => {
                p.with(&["files", "count"], |p| expr(&f.count, p));
                p.with(&["files", "size"], |p| distref(&f.size, p));
                if let Some(e) = &f.samples_per_file {
                    p.with(&["files", "samples_per_file"], |p| expr(e, p));
                }
                if let Some(e) = &f.chunk {
                    p.with(&["files", "chunk"], |p| expr(e, p));
                }
            }
            Dataset::Regions(r) => {
                p.with(&["regions", "count"], |p| expr(&r.count, p));
                p.with(&["regions", "slot"], |p| expr(&r.slot, p));
                p.with(&["regions", "size"], |p| distref(&r.size, p));
            }
        });
    }
    for (name, ns) in &ast.namespaces {
        if let NsSize::Expr(e) = &ns.size {
            p.with(&["namespaces", name, "size"], |p| expr(e, p));
        }
    }
    for (name, actor) in &ast.actors {
        p.with(&["actors", name], |p| {
            if let Some(c) = &actor.count {
                p.with(&["count"], |p| expr(c, p));
            }
            p.with(&["body"], |p| body(&actor.body, p));
        });
    }
}

#[derive(Default)]
struct Path {
    parts: Vec<String>,
}

impl Path {
    fn with<R>(&mut self, segs: &[&str], f: impl FnOnce(&mut Path) -> R) -> R {
        for s in segs {
            self.parts.push(escape(s));
        }
        let r = f(self);
        for _ in segs {
            self.parts.pop();
        }
        r
    }

    fn pointer(&self) -> String {
        let mut s = String::new();
        for p in &self.parts {
            s.push('/');
            s.push_str(p);
        }
        s
    }
}

fn escape(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

fn pvalue(v: &PValue, p: &mut Path) {
    match v {
        PValue::Dist(d) => dist(d, p),
        PValue::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                p.with(&[&i.to_string()], |p| pvalue(x, p));
            }
        }
        PValue::Scalar(_) => {}
    }
}

fn dist(d: &Dist, p: &mut Path) {
    match d {
        Dist::Const(e) => p.with(&["const"], |p| expr(e, p)),
        Dist::Uniform { lo, hi } => {
            p.with(&["uniform", "lo"], |p| expr(lo, p));
            p.with(&["uniform", "hi"], |p| expr(hi, p));
        }
        Dist::Uniform64 {} | Dist::Zipf { .. } | Dist::Hotset { .. } | Dist::Empirical { .. } => {}
        Dist::Normal { mean, sd, min, max } => {
            p.with(&["normal", "mean"], |p| expr(mean, p));
            p.with(&["normal", "sd"], |p| expr(sd, p));
            if let Some(e) = min {
                p.with(&["normal", "min"], |p| expr(e, p));
            }
            if let Some(e) = max {
                p.with(&["normal", "max"], |p| expr(e, p));
            }
        }
        Dist::Lognormal { median, min, max, .. } => {
            p.with(&["lognormal", "median"], |p| expr(median, p));
            if let Some(e) = min {
                p.with(&["lognormal", "min"], |p| expr(e, p));
            }
            if let Some(e) = max {
                p.with(&["lognormal", "max"], |p| expr(e, p));
            }
        }
        Dist::Mixture(arms) => {
            for (i, arm) in arms.iter().enumerate() {
                if let Some(d) = &arm.dist {
                    p.with(&["mixture", &i.to_string(), "dist"], |p| distref(d, p));
                }
            }
        }
    }
}

fn distref(d: &DistRef, p: &mut Path) {
    match d {
        DistRef::Dist(d) => dist(d, p),
        DistRef::Expr(e) => expr(e, p),
    }
}

fn exprlike(v: &ExprLike, p: &mut Path) {
    match v {
        ExprLike::Expr(e) => expr(e, p),
        ExprLike::Dist(d) => dist(d, p),
        ExprLike::Handle(h) => handle(h, p),
    }
}

fn expr(e: &Expr, p: &mut Path) {
    let node = match e {
        Expr::Lit(_) => return,
        Expr::Node(n) => n.as_ref(),
    };
    match node {
        ExprNode::Param(_) | ExprNode::Index(_) | ExprNode::Ref(_) | ExprNode::Actor(_) | ExprNode::Len(_) | ExprNode::Sum(_) | ExprNode::Count(_) | ExprNode::Dirs(_) => {}
        ExprNode::At { index, .. } => p.with(&["at", "index"], |p| expr(index, p)),
        ExprNode::Draw(d) => {
            d.site.set(site_hash(&p.pointer()));
            p.with(&["draw"], |p| distref(&d.dist, p));
        }
        ExprNode::Elem { index, .. } => p.with(&["elem", "index"], |p| expr(index, p)),
        ExprNode::Size(h) => p.with(&["size"], |p| handle(h, p)),
        ExprNode::Offset(h) => p.with(&["offset"], |p| handle(h, p)),
        ExprNode::UnitIndex(h) => p.with(&["unit_index"], |p| handle(h, p)),
        ExprNode::Units(h) => p.with(&["units"], |p| handle(h, p)),
        ExprNode::Chunks(h) => p.with(&["chunks"], |p| handle(h, p)),
        ExprNode::Add(a) => args(a, "add", p),
        ExprNode::Sub(a) => args(a, "sub", p),
        ExprNode::Mul(a) => args(a, "mul", p),
        ExprNode::Div(a) => args(a, "div", p),
        ExprNode::Mod(a) => args(a, "mod", p),
        ExprNode::CeilDiv(a) => args(a, "ceil_div", p),
        ExprNode::Min(a) => args(a, "min", p),
        ExprNode::Max(a) => args(a, "max", p),
        ExprNode::Eq(a) => args(a, "eq", p),
        ExprNode::Ne(a) => args(a, "ne", p),
        ExprNode::Lt(a) => args(a, "lt", p),
        ExprNode::Le(a) => args(a, "le", p),
        ExprNode::Gt(a) => args(a, "gt", p),
        ExprNode::Ge(a) => args(a, "ge", p),
        ExprNode::And(a) => args(a, "and", p),
        ExprNode::Or(a) => args(a, "or", p),
        ExprNode::Neg(e) => p.with(&["neg"], |p| expr(e, p)),
        ExprNode::Not(e) => p.with(&["not"], |p| expr(e, p)),
        ExprNode::Cond(c) => {
            p.with(&["cond", "if"], |p| expr(&c.test, p));
            p.with(&["cond", "then"], |p| exprlike(&c.then, p));
            p.with(&["cond", "else"], |p| exprlike(&c.otherwise, p));
        }
    }
}

fn args(a: &[Expr], key: &str, p: &mut Path) {
    for (i, e) in a.iter().enumerate() {
        p.with(&[key, &i.to_string()], |p| expr(e, p));
    }
}

fn handle(h: &Handle, p: &mut Path) {
    match h {
        Handle::Ref(_) | Handle::Consume(_) => {}
        Handle::File(f) => {
            if let Some(e) = &f.id {
                p.with(&["file", "id"], |p| expr(e, p));
            }
            if let Some(of) = &f.of {
                p.with(&["file", "of"], |p| handle(of, p));
            }
            if let Some(e) = &f.chunk {
                p.with(&["file", "chunk"], |p| expr(e, p));
            }
        }
        Handle::Dir { id, .. } => p.with(&["dir", "id"], |p| expr(id, p)),
        Handle::Object { fields, .. } => {
            for (name, e) in fields {
                p.with(&["object", "fields", name], |p| expr(e, p));
            }
        }
        Handle::Pick(pick) => {
            pick.site.set(site_hash(&p.pointer()));
            if let Some(d) = &pick.dist {
                p.with(&["pick", "dist"], |p| distref(d, p));
            }
        }
        Handle::Unit(u) => {
            p.with(&["unit", "of"], |p| handle(&u.of, p));
            if let Some(e) = &u.index {
                p.with(&["unit", "index"], |p| expr(e, p));
            }
        }
        Handle::Column(c) => {
            p.with(&["column", "of"], |p| handle(&c.of, p));
            p.with(&["column", "index"], |p| expr(&c.index, p));
        }
    }
}

fn body(nodes: &[Node], p: &mut Path) {
    for (i, n) in nodes.iter().enumerate() {
        p.with(&[&i.to_string()], |p| node(n, p));
    }
}

fn node(n: &Node, p: &mut Path) {
    let k = n.kind();
    match n {
        Node::Let { value, .. } => p.with(&[k, "value"], |p| exprlike(value, p)),
        Node::Loop { from, to, step, body: b, .. } => {
            if let Some(e) = from {
                p.with(&[k, "from"], |p| expr(e, p));
            }
            p.with(&[k, "to"], |p| expr(to, p));
            if let Some(e) = step {
                p.with(&[k, "step"], |p| expr(e, p));
            }
            p.with(&[k, "body"], |p| body(b, p));
        }
        Node::Parallel { width, body: b, .. } => {
            p.with(&[k, "width"], |p| expr(width, p));
            p.with(&[k, "body"], |p| body(b, p));
        }
        Node::Channel { capacity, .. } => p.with(&[k, "capacity"], |p| expr(capacity, p)),
        Node::Put { seq, .. } => p.with(&[k, "seq"], |p| expr(seq, p)),
        Node::Take { .. } | Node::Barrier { .. } | Node::Trace { .. } => {}
        Node::Loader { workers, prefetch, batches, body: b, .. } => {
            p.with(&[k, "workers"], |p| expr(workers, p));
            p.with(&[k, "prefetch"], |p| expr(prefetch, p));
            p.with(&[k, "batches"], |p| expr(batches, p));
            p.with(&[k, "body"], |p| body(b, p));
        }
        Node::Compute { ns } => p.with(&[k, "ns"], |p| expr(ns, p)),
        Node::Cond { test, then, otherwise } => {
            p.with(&[k, "if"], |p| expr(test, p));
            p.with(&[k, "then"], |p| body(then, p));
            if let Some(b) = otherwise {
                p.with(&[k, "else"], |p| body(b, p));
            }
        }
        Node::Choose { arms, site } => {
            site.set(site_hash(&p.pointer()));
            for (i, arm) in arms.iter().enumerate() {
                p.with(&[k, "arms", &i.to_string(), "body"], |p| body(&arm.body, p));
            }
        }
        Node::Phase { name, body: b } => {
            p.with(&[k, "name"], |p| expr(name, p));
            p.with(&[k, "body"], |p| body(b, p));
        }
        Node::Open { file, .. } => p.with(&[k, "file"], |p| handle(file, p)),
        Node::Close(f) | Node::Fstat(f) | Node::Stat(f) | Node::Fsync(f) | Node::Fdatasync(f) | Node::Unlink(f) => {
            p.with(&[k, "file"], |p| handle(&f.file, p))
        }
        Node::Read { file, len, offset, repeat, .. } => {
            p.with(&[k, "file"], |p| handle(file, p));
            p.with(&[k, "len"], |p| expr(len, p));
            if let Some(e) = offset {
                p.with(&[k, "offset"], |p| expr(e, p));
            }
            if let Some(Repeat::Count(e)) = repeat {
                p.with(&[k, "repeat"], |p| expr(e, p));
            }
        }
        Node::Write { file, len, offset, repeat, .. } => {
            p.with(&[k, "file"], |p| handle(file, p));
            p.with(&[k, "len"], |p| expr(len, p));
            if let Some(e) = offset {
                p.with(&[k, "offset"], |p| expr(e, p));
            }
            if let Some(e) = repeat {
                p.with(&[k, "repeat"], |p| expr(e, p));
            }
        }
        Node::Lseek { file, offset, .. } => {
            p.with(&[k, "file"], |p| handle(file, p));
            p.with(&[k, "offset"], |p| expr(offset, p));
        }
        Node::Ioctl { file, .. } => p.with(&[k, "file"], |p| handle(file, p)),
        Node::Fadvise { file, offset, len, .. } => {
            p.with(&[k, "file"], |p| handle(file, p));
            if let Some(e) = offset {
                p.with(&[k, "offset"], |p| expr(e, p));
            }
            if let Some(e) = len {
                p.with(&[k, "len"], |p| expr(e, p));
            }
        }
        Node::Ftruncate { file, len, .. } => {
            p.with(&[k, "file"], |p| handle(file, p));
            p.with(&[k, "len"], |p| expr(len, p));
        }
        Node::Fallocate { file, offset, len, .. } => {
            p.with(&[k, "file"], |p| handle(file, p));
            if let Some(e) = offset {
                p.with(&[k, "offset"], |p| expr(e, p));
            }
            p.with(&[k, "len"], |p| expr(len, p));
        }
        Node::Mkdir { dir, .. } | Node::Rmdir { dir, .. } | Node::Readdir { dir, .. } => p.with(&[k, "dir"], |p| handle(dir, p)),
        Node::Rename { from, to, .. } => {
            p.with(&[k, "from"], |p| handle(from, p));
            p.with(&[k, "to"], |p| handle(to, p));
        }
    }
}
