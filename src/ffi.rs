//! C ABI surface — everything declared in include/embedkit.h.
//! Conventions (house standard): string inputs are const char* UTF-8;
//! `*mut c_char` results are caller-owned (free with ek_free_string);
//! failures return an ERR_* code (or null / 0 / -1) + thread-local
//! last error; nothing unwinds across the boundary.
//!
//! Handles are opaque u64s into two process-wide registries (the
//! FastSearch/Whisper pattern): the registry lock is held only to
//! clone the per-object `Arc<Mutex<_>>` out, never during work, so
//! distinct handles operate in parallel while calls on one handle
//! serialize on its own mutex.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::embedder::Embedder;
use crate::index::{hits_json, Index};
use crate::{io_error, CResult, EK_OK, ERR_INTERNAL, ERR_INVALID_ARG, ERR_NOT_FOUND};

thread_local! {
    static LAST_ERROR: RefCell<(i32, CString)> =
        RefCell::new((EK_OK, CString::new("").unwrap()));
}

fn set_error(code: i32, msg: &str) {
    let clean: String = msg.chars().filter(|&c| c != '\0').collect();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = (
            code,
            CString::new(clean).unwrap_or_else(|_| CString::new("error").unwrap()),
        );
    });
}

fn clear_error() {
    set_error(EK_OK, "");
}

fn guard<T, F: FnOnce() -> T>(f: F) -> std::thread::Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
}

fn input(ptr: *const c_char, what: &str) -> Option<String> {
    if ptr.is_null() {
        set_error(ERR_INVALID_ARG, &format!("{what} must not be null"));
        return None;
    }
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(s) => Some(s.to_string()),
        Err(_) => {
            set_error(ERR_INVALID_ARG, &format!("{what} is not valid UTF-8"));
            None
        }
    }
}

/// Optional string input: null stays None (e.g. entry text).
fn opt_input(ptr: *const c_char, what: &str) -> Option<Option<String>> {
    if ptr.is_null() {
        Some(None)
    } else {
        input(ptr, what).map(Some)
    }
}

/// Move a Rust String across the boundary as a caller-owned C string.
fn string_out(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => {
            clear_error();
            c.into_raw()
        }
        Err(_) => {
            set_error(ERR_INTERNAL, "result contained an interior NUL byte");
            std::ptr::null_mut()
        }
    }
}

fn fail<T>(r: CResult<T>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err((c, m)) => {
            set_error(c, &m);
            None
        }
    }
}

fn code(r: CResult<()>) -> c_int {
    match r {
        Ok(()) => {
            clear_error();
            EK_OK as c_int
        }
        Err((c, m)) => {
            set_error(c, &m);
            c as c_int
        }
    }
}

// ---------------------------------------------------------------------------
// Registries
// ---------------------------------------------------------------------------

fn index_registry() -> &'static Mutex<HashMap<u64, Arc<Mutex<Index>>>> {
    static REG: OnceLock<Mutex<HashMap<u64, Arc<Mutex<Index>>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedder_registry() -> &'static Mutex<HashMap<u64, Arc<Mutex<Embedder>>>> {
    static REG: OnceLock<Mutex<HashMap<u64, Arc<Mutex<Embedder>>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn register_index(ix: Index) -> u64 {
    let h = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
    index_registry()
        .lock()
        .unwrap()
        .insert(h, Arc::new(Mutex::new(ix)));
    h
}

fn lookup_index(handle: u64) -> Option<Arc<Mutex<Index>>> {
    index_registry().lock().unwrap().get(&handle).cloned()
}

fn lookup_embedder(handle: u64) -> Option<Arc<Mutex<Embedder>>> {
    embedder_registry().lock().unwrap().get(&handle).cloned()
}

fn no_index(handle: u64) -> (i32, String) {
    (ERR_NOT_FOUND, format!("no such index handle: {handle}"))
}

fn no_embedder(handle: u64) -> (i32, String) {
    (ERR_NOT_FOUND, format!("no such embedder handle: {handle}"))
}

// ---------------------------------------------------------------------------
// Version & errors
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ek_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn ek_version_major() -> u32 {
    crate::VERSION_MAJOR
}

#[no_mangle]
pub extern "C" fn ek_version_minor() -> u32 {
    crate::VERSION_MINOR
}

#[no_mangle]
pub extern "C" fn ek_version_patch() -> u32 {
    crate::VERSION_PATCH
}

#[no_mangle]
pub extern "C" fn ek_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().1.as_ptr())
}

#[no_mangle]
pub extern "C" fn ek_last_error_code() -> c_int {
    LAST_ERROR.with(|e| e.borrow().0)
}

#[no_mangle]
pub extern "C" fn ek_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = guard(|| unsafe {
            drop(CString::from_raw(ptr));
        });
    }
}

// ---------------------------------------------------------------------------
// Index lifecycle
// ---------------------------------------------------------------------------

/// Create an empty index with the given (fixed) dimension.
/// Returns a handle (never 0), or 0 on failure.
#[no_mangle]
pub extern "C" fn ek_index_create(dim: u32) -> u64 {
    match guard(|| match fail(Index::new(dim as usize)) {
        Some(ix) => {
            clear_error();
            register_index(ix)
        }
        None => 0,
    }) {
        Ok(h) => h,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            0
        }
    }
}

/// Free an index. EK_OK, or EK_ERR_NOT_FOUND for a bad handle.
#[no_mangle]
pub extern "C" fn ek_index_free(handle: u64) -> c_int {
    match guard(|| {
        if index_registry().lock().unwrap().remove(&handle).is_some() {
            EK_OK as c_int
        } else {
            let (c, m) = no_index(handle);
            set_error(c, &m);
            c as c_int
        }
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL as c_int
        }
    }
}

/// Number of entries in an index, or -1 on error.
#[no_mangle]
pub extern "C" fn ek_index_count(handle: u64) -> i64 {
    match guard(|| match lookup_index(handle) {
        Some(ix) => {
            clear_error();
            let n = ix.lock().unwrap().len() as i64;
            n
        }
        None => {
            let (c, m) = no_index(handle);
            set_error(c, &m);
            -1
        }
    }) {
        Ok(n) => n,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Index mutation & search
// ---------------------------------------------------------------------------

/// Add (or replace) an entry with a caller-supplied vector.
/// `text` may be null; it is stored and echoed in search results.
/// `vector` must point to `dim` readable f32 values and `dim` must
/// equal the index dimension.
#[no_mangle]
pub extern "C" fn ek_index_add(
    handle: u64,
    id: *const c_char,
    text: *const c_char,
    vector: *const f32,
    dim: u32,
) -> c_int {
    match guard(|| {
        let Some(id) = input(id, "id") else {
            return ek_last_error_code();
        };
        let Some(text) = opt_input(text, "text") else {
            return ek_last_error_code();
        };
        if vector.is_null() {
            set_error(ERR_INVALID_ARG, "vector must not be null");
            return ERR_INVALID_ARG as c_int;
        }
        let vec: Vec<f32> = unsafe { std::slice::from_raw_parts(vector, dim as usize) }.to_vec();
        let Some(ix) = lookup_index(handle) else {
            let (c, m) = no_index(handle);
            set_error(c, &m);
            return c as c_int;
        };
        let r = ix.lock().unwrap().add(&id, text, vec);
        code(r)
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL as c_int
        }
    }
}

/// Embed `text` with `embedder` and add it to the index under `id`
/// (replacing any existing entry with that id). The text itself is
/// stored as the entry's text.
#[no_mangle]
pub extern "C" fn ek_index_add_text(
    index: u64,
    embedder: u64,
    id: *const c_char,
    text: *const c_char,
) -> c_int {
    match guard(|| {
        let Some(id) = input(id, "id") else {
            return ek_last_error_code();
        };
        let Some(text) = input(text, "text") else {
            return ek_last_error_code();
        };
        match add_text_impl(index, embedder, &id, &text) {
            Ok(()) => {
                clear_error();
                EK_OK as c_int
            }
            Err((c, m)) => {
                set_error(c, &m);
                c as c_int
            }
        }
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL as c_int
        }
    }
}

fn add_text_impl(index: u64, embedder: u64, id: &str, text: &str) -> CResult<()> {
    let emb = lookup_embedder(embedder).ok_or_else(|| no_embedder(embedder))?;
    let ix = lookup_index(index).ok_or_else(|| no_index(index))?;
    // Lock order is always embedder → index (here and in
    // search_text); no path takes them in the opposite order.
    let vector = emb.lock().unwrap().embed(text)?;
    let r = ix.lock().unwrap().add(id, Some(text.to_string()), vector);
    r
}

/// Remove an entry by id. EK_OK, or EK_ERR_NOT_FOUND.
#[no_mangle]
pub extern "C" fn ek_index_remove(handle: u64, id: *const c_char) -> c_int {
    match guard(|| {
        let Some(id) = input(id, "id") else {
            return ek_last_error_code();
        };
        let Some(ix) = lookup_index(handle) else {
            let (c, m) = no_index(handle);
            set_error(c, &m);
            return c as c_int;
        };
        let r = ix.lock().unwrap().remove(&id);
        code(r)
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL as c_int
        }
    }
}

/// Exact cosine top-k search with a caller-supplied query vector.
/// Returns a caller-owned JSON array:
/// `[{"id": "...", "score": 0.83, "text": "..."}]` (the text key is
/// omitted for entries stored without text), or null on failure.
/// `k <= 0` returns every entry, ranked.
#[no_mangle]
pub extern "C" fn ek_search(handle: u64, query: *const f32, dim: u32, k: c_int) -> *mut c_char {
    match guard(|| {
        if query.is_null() {
            set_error(ERR_INVALID_ARG, "query vector must not be null");
            return std::ptr::null_mut();
        }
        let q: Vec<f32> = unsafe { std::slice::from_raw_parts(query, dim as usize) }.to_vec();
        let Some(ix) = lookup_index(handle) else {
            let (c, m) = no_index(handle);
            set_error(c, &m);
            return std::ptr::null_mut();
        };
        let kk = if k <= 0 { 0 } else { k as usize };
        let r = ix.lock().unwrap().search(&q, kk);
        match r {
            Ok(hits) => string_out(hits_json(&hits)),
            Err((c, m)) => {
                set_error(c, &m);
                std::ptr::null_mut()
            }
        }
    }) {
        Ok(p) => p,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            std::ptr::null_mut()
        }
    }
}

/// Embed `query_text` with `embedder` and search the index with it.
/// Same JSON shape and `k` semantics as ek_search.
#[no_mangle]
pub extern "C" fn ek_search_text(
    index: u64,
    embedder: u64,
    query_text: *const c_char,
    k: c_int,
) -> *mut c_char {
    match guard(|| {
        let Some(text) = input(query_text, "query text") else {
            return std::ptr::null_mut();
        };
        let r = (|| -> CResult<String> {
            let emb = lookup_embedder(embedder).ok_or_else(|| no_embedder(embedder))?;
            let ix = lookup_index(index).ok_or_else(|| no_index(index))?;
            let vector = emb.lock().unwrap().embed(&text)?;
            let kk = if k <= 0 { 0 } else { k as usize };
            let hits = ix.lock().unwrap().search(&vector, kk)?;
            Ok(hits_json(&hits))
        })();
        match r {
            Ok(j) => string_out(j),
            Err((c, m)) => {
                set_error(c, &m);
                std::ptr::null_mut()
            }
        }
    }) {
        Ok(p) => p,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Index persistence (EKIDX1)
// ---------------------------------------------------------------------------

/// Save an index to `path` in the EKIDX1 format (see embedkit.h).
#[no_mangle]
pub extern "C" fn ek_index_save(handle: u64, path: *const c_char) -> c_int {
    match guard(|| {
        let Some(path) = input(path, "path") else {
            return ek_last_error_code();
        };
        let Some(ix) = lookup_index(handle) else {
            let (c, m) = no_index(handle);
            set_error(c, &m);
            return c as c_int;
        };
        let bytes = ix.lock().unwrap().save_to_bytes();
        code(std::fs::write(&path, bytes).map_err(|e| io_error("write", &path, &e)))
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL as c_int
        }
    }
}

/// Load an EKIDX1 file as a new index; returns its handle, or 0 on
/// failure (missing file → EK_ERR_NOT_FOUND, corrupt → EK_ERR_PLATFORM).
#[no_mangle]
pub extern "C" fn ek_index_load(path: *const c_char) -> u64 {
    match guard(|| {
        let Some(path) = input(path, "path") else {
            return 0;
        };
        let r = (|| -> CResult<Index> {
            let bytes = std::fs::read(&path).map_err(|e| io_error("read", &path, &e))?;
            Index::load_from_bytes(&bytes)
        })();
        match fail(r) {
            Some(ix) => {
                clear_error();
                register_index(ix)
            }
            None => 0,
        }
    }) {
        Ok(h) => h,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Embedder lifecycle
// ---------------------------------------------------------------------------

/// Create an embedder: load the ONNX Runtime library at
/// `ort_lib_path`, open the model at `model_path`, read the
/// WordPiece vocab at `vocab_path`. `max_len` caps the tokenized
/// sequence length; 0 selects the default (256). Returns a handle
/// (never 0), or 0 on failure — missing files are EK_ERR_NOT_FOUND,
/// an old/unloadable runtime or a non-BERT model shape is reported
/// precisely in ek_last_error.
#[no_mangle]
pub extern "C" fn ek_embedder_create(
    ort_lib_path: *const c_char,
    model_path: *const c_char,
    vocab_path: *const c_char,
    max_len: u32,
) -> u64 {
    match guard(|| {
        let Some(lib) = input(ort_lib_path, "ONNX Runtime library path") else {
            return 0;
        };
        let Some(model) = input(model_path, "model path") else {
            return 0;
        };
        let Some(vocab) = input(vocab_path, "vocab path") else {
            return 0;
        };
        match fail(Embedder::create(&lib, &model, &vocab, max_len as usize)) {
            Some(emb) => {
                let h = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
                embedder_registry()
                    .lock()
                    .unwrap()
                    .insert(h, Arc::new(Mutex::new(emb)));
                clear_error();
                h
            }
            None => 0,
        }
    }) {
        Ok(h) => h,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            0
        }
    }
}

/// Free an embedder (releasing its session and runtime).
/// EK_OK, or EK_ERR_NOT_FOUND for a bad handle.
#[no_mangle]
pub extern "C" fn ek_embedder_free(handle: u64) -> c_int {
    match guard(|| {
        if embedder_registry()
            .lock()
            .unwrap()
            .remove(&handle)
            .is_some()
        {
            EK_OK as c_int
        } else {
            let (c, m) = no_embedder(handle);
            set_error(c, &m);
            c as c_int
        }
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL as c_int
        }
    }
}

/// The embedding dimension of an embedder's model, or -1 on error.
#[no_mangle]
pub extern "C" fn ek_embedder_dim(handle: u64) -> i64 {
    match guard(|| match lookup_embedder(handle) {
        Some(emb) => {
            clear_error();
            let d = emb.lock().unwrap().dim() as i64;
            d
        }
        None => {
            let (c, m) = no_embedder(handle);
            set_error(c, &m);
            -1
        }
    }) {
        Ok(n) => n,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            -1
        }
    }
}

/// Embed one text. Returns a caller-owned JSON string
/// `{"dim": 384, "vector": [0.0123, ...]}`, or null on failure.
/// The vector is mean-pooled and L2-normalized.
#[no_mangle]
pub extern "C" fn ek_embed(handle: u64, text: *const c_char) -> *mut c_char {
    match guard(|| {
        let Some(text) = input(text, "text") else {
            return std::ptr::null_mut();
        };
        let r = (|| -> CResult<String> {
            let emb = lookup_embedder(handle).ok_or_else(|| no_embedder(handle))?;
            let v = emb.lock().unwrap().embed(&text)?;
            let mut out = String::with_capacity(v.len() * 10 + 24);
            out.push_str("{\"dim\": ");
            out.push_str(&v.len().to_string());
            out.push_str(", \"vector\": [");
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&crate::f32_json(*x));
            }
            out.push_str("]}");
            Ok(out)
        })();
        match r {
            Ok(j) => string_out(j),
            Err((c, m)) => {
                set_error(c, &m);
                std::ptr::null_mut()
            }
        }
    }) {
        Ok(p) => p,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn cs(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    fn take(p: *mut c_char) -> String {
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
        ek_free_string(p);
        s
    }

    #[test]
    fn ffi_contract() {
        assert!(!ek_version().is_null());
        assert_eq!(ek_version_major(), 0);
        assert_eq!(ek_version_minor(), 1);
        assert_eq!(ek_version_patch(), 0);

        // dim 0 is invalid.
        assert_eq!(ek_index_create(0), 0);
        assert_eq!(ek_last_error_code(), ERR_INVALID_ARG);

        let ix = ek_index_create(2);
        assert!(ix != 0);
        assert_eq!(ek_index_count(ix), 0);

        let v = [1.0f32, 0.0];
        assert_eq!(
            ek_index_add(
                ix,
                cs("east").as_ptr(),
                cs("pointing east").as_ptr(),
                v.as_ptr(),
                2
            ),
            EK_OK as c_int
        );
        // Dim mismatch and null vector are INVALID_ARG.
        let v3 = [1.0f32, 0.0, 0.0];
        assert_eq!(
            ek_index_add(ix, cs("bad").as_ptr(), ptr::null(), v3.as_ptr(), 3),
            ERR_INVALID_ARG as c_int
        );
        assert_eq!(
            ek_index_add(ix, cs("bad").as_ptr(), ptr::null(), ptr::null(), 2),
            ERR_INVALID_ARG as c_int
        );
        assert_eq!(ek_index_count(ix), 1);

        let j = take(ek_search(ix, v.as_ptr(), 2, 1));
        assert!(j.contains("\"id\": \"east\""), "{j}");
        assert!(j.contains("\"text\": \"pointing east\""), "{j}");

        // Search with a wrong-dim query → null + INVALID_ARG.
        assert!(ek_search(ix, v3.as_ptr(), 3, 1).is_null());
        assert_eq!(ek_last_error_code(), ERR_INVALID_ARG);

        assert_eq!(ek_index_remove(ix, cs("east").as_ptr()), EK_OK as c_int);
        assert_eq!(
            ek_index_remove(ix, cs("east").as_ptr()),
            ERR_NOT_FOUND as c_int
        );

        // Bad handles everywhere.
        assert_eq!(ek_index_count(999_999), -1);
        assert_eq!(ek_last_error_code(), ERR_NOT_FOUND);
        assert_eq!(ek_index_free(999_999), ERR_NOT_FOUND as c_int);
        assert_eq!(ek_embedder_dim(999_999), -1);
        assert!(ek_embed(999_999, cs("x").as_ptr()).is_null());
        assert_eq!(ek_last_error_code(), ERR_NOT_FOUND);

        assert_eq!(ek_index_free(ix), EK_OK as c_int);
        assert_eq!(ek_index_free(ix), ERR_NOT_FOUND as c_int);
        ek_free_string(ptr::null_mut());
    }

    #[test]
    fn ffi_save_load_roundtrip() {
        let ix = ek_index_create(2);
        let v = [0.6f32, 0.8];
        assert_eq!(
            ek_index_add(ix, cs("d").as_ptr(), ptr::null(), v.as_ptr(), 2),
            EK_OK as c_int
        );
        let path = std::env::temp_dir().join(format!("ek-ffi-test-{}.ekidx", std::process::id()));
        let p = cs(path.to_str().unwrap());
        assert_eq!(ek_index_save(ix, p.as_ptr()), EK_OK as c_int);
        let ix2 = ek_index_load(p.as_ptr());
        assert!(ix2 != 0);
        assert_eq!(ek_index_count(ix2), 1);
        assert_eq!(ek_index_free(ix), EK_OK as c_int);
        assert_eq!(ek_index_free(ix2), EK_OK as c_int);
        let _ = std::fs::remove_file(&path);

        // Missing file → 0 + NOT_FOUND.
        assert_eq!(ek_index_load(cs("no/such/file.ekidx").as_ptr()), 0);
        assert_eq!(ek_last_error_code(), ERR_NOT_FOUND);
    }

    #[test]
    fn embedder_missing_files() {
        assert_eq!(
            ek_embedder_create(
                cs("no/lib.so").as_ptr(),
                cs("no/model.onnx").as_ptr(),
                cs("no/vocab.txt").as_ptr(),
                0,
            ),
            0
        );
        assert_eq!(ek_last_error_code(), ERR_NOT_FOUND);
        // Null path → INVALID_ARG.
        assert_eq!(
            ek_embedder_create(ptr::null(), cs("m").as_ptr(), cs("v").as_ptr(), 0),
            0
        );
        assert_eq!(ek_last_error_code(), ERR_INVALID_ARG);
    }
}
