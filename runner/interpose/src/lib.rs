//! The interposition machinery of aeiou's `LD_PRELOAD` libraries (`DESIGN_REVIEW.md` §3.96,
//! brief items 54 and 55): one crate, a library per use, so the capture's library, which only
//! observes, holds no code that changes what the application sees.
//!
//! - `next`: the real definition of a symbol this library also defines, found past a library
//!   loaded `RTLD_LOCAL` (CPython loads `_ssl` and its `libssl` so, and `dlsym(RTLD_NEXT)`
//!   returns NULL for them).
//! - `Guard`: a thread's way into a hook once; a call the hook itself makes (a `write` inside
//!   `SSL_write`, an `open` of the hook's own) passes straight through.
//! - `log`: an appending log in a file every process of the application maps, written with no
//!   system call (a `write` would be in the application's trace), read after the run.
//! - `capture`: the capture library's records in that log, the one definition its writer and
//!   `aeiou-trace capture`, its reader, share.
use std::cell::Cell;
use std::ffi::{c_void, CStr};

pub mod capture;
pub mod log;

/// The definition of `name` that a call would reach without this library: the next in the
/// lookup order, else any loaded object's but `own` (this library's, by its address).
pub fn next(name: &CStr, own: *const c_void) -> *mut c_void {
    let s = unsafe { libc::dlsym(libc::RTLD_NEXT, name.as_ptr()) };
    if !s.is_null() && s as *const c_void != own {
        return s;
    }
    struct Find<'a> {
        name: &'a CStr,
        own: *const c_void,
        hit: *mut c_void,
    }
    unsafe extern "C" fn one(info: *mut libc::dl_phdr_info, _: usize, data: *mut c_void) -> libc::c_int {
        let f = &mut *(data as *mut Find);
        let n = (*info).dlpi_name;
        if n.is_null() || *n == 0 {
            return 0;
        }
        let h = libc::dlopen(n, libc::RTLD_LAZY | libc::RTLD_NOLOAD);
        if h.is_null() {
            return 0;
        }
        let s = libc::dlsym(h, f.name.as_ptr());
        libc::dlclose(h);
        if !s.is_null() && s as *const c_void != f.own {
            f.hit = s;
            return 1;
        }
        0
    }
    let mut f = Find { name, own, hit: std::ptr::null_mut() };
    unsafe { libc::dl_iterate_phdr(Some(one), &mut f as *mut Find as *mut c_void) };
    f.hit
}

thread_local! {
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// Held while a thread is inside a hook; dropped on the way out.
pub struct Guard(());

impl Guard {
    /// The way in, unless this thread is in a hook already.
    pub fn enter() -> Option<Guard> {
        BUSY.with(|b| (!b.replace(true)).then_some(Guard(())))
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        BUSY.with(|b| b.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_enters_once_and_again_after_leaving() {
        let g = Guard::enter().expect("first");
        assert!(Guard::enter().is_none());
        // another thread is not this one
        std::thread::spawn(|| assert!(Guard::enter().is_some())).join().unwrap();
        drop(g);
        assert!(Guard::enter().is_some());
    }

    #[test]
    fn the_next_definition_is_libc_s_and_a_missing_name_is_null() {
        let own = the_next_definition_is_libc_s_and_a_missing_name_is_null as *const c_void;
        let w = next(c"getppid", own);
        assert!(!w.is_null());
        let f: extern "C" fn() -> libc::pid_t = unsafe { std::mem::transmute(w) };
        assert_eq!(f(), unsafe { libc::getppid() });
        assert!(next(c"aeiou_no_such_symbol", own).is_null());
        // asked past libc's own definition: the walk of the loaded objects skips it, and no
        // other object defines it
        assert!(next(c"getppid", w).is_null());
    }
}
