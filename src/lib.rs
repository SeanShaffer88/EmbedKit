//! EmbedKit — pure logic core: error convention, JSON rendering.
//! The vector index lives in [`index`], the BERT WordPiece tokenizer
//! in [`tokenizer`], the runtime-loaded ONNX Runtime binding in
//! [`ort`], the model embedder in [`embedder`], and the C ABI in
//! [`ffi`].
//!
//! Design: two layers behind one ABI. Layer 1 is an exact (brute
//! force) cosine-similarity index over caller-supplied vectors —
//! pure Rust, no model, always available. Layer 2 turns text into
//! vectors with a BERT-class sentence model (all-MiniLM shape) run
//! by ONNX Runtime, which the embedder loads *at runtime* from a
//! caller-supplied library path: EmbedKit never links it. When no
//! runtime/model is present, embedder calls report
//! [`ERR_NOT_SUPPORTED`] precisely and layer 1 keeps working —
//! the same honest contract as the sibling DLLs in the collection.
//! Everything is local: no network, no telemetry.

pub mod embedder;
pub mod ffi;
pub mod index;
pub mod ort;
pub mod tokenizer;

pub const VERSION: &str = "0.1.0";
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

// House error convention (win-common 0–6), codes returned by the ABI
// and stored alongside the thread-local last-error message.
pub const EK_OK: i32 = 0;
pub const ERR_INVALID_ARG: i32 = 1;
pub const ERR_NOT_SUPPORTED: i32 = 2;
pub const ERR_NOT_FOUND: i32 = 3;
pub const ERR_ACCESS_DENIED: i32 = 4;
pub const ERR_PLATFORM: i32 = 5;
pub const ERR_INTERNAL: i32 = 6;

pub type CResult<T> = Result<T, (i32, String)>;

/// Map a filesystem error to the house codes: a missing file is
/// NOT_FOUND and a permission failure is ACCESS_DENIED, never a
/// misleading generic error.
pub fn io_error(what: &str, path: &str, e: &std::io::Error) -> (i32, String) {
    let code = match e.kind() {
        std::io::ErrorKind::NotFound => ERR_NOT_FOUND,
        std::io::ErrorKind::PermissionDenied => ERR_ACCESS_DENIED,
        _ => ERR_PLATFORM,
    };
    (code, format!("cannot {what} '{path}': {e}"))
}

/// Require a path to name an existing file, with the precise code.
pub fn require_file(path: &str, what: &str) -> CResult<()> {
    if std::path::Path::new(path).is_file() {
        Ok(())
    } else {
        Err((ERR_NOT_FOUND, format!("{what} not found: '{path}'")))
    }
}

pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Shortest round-trip rendering of an f32 (Rust's Display for f32),
/// so a vector printed into JSON parses back to the identical bits.
pub fn f32_json(v: f32) -> String {
    format!("{v}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes() {
        assert_eq!(json_escape("a\"b\\c\nd\u{1}"), "a\\\"b\\\\c\\nd\\u0001");
    }

    #[test]
    fn f32_json_roundtrips() {
        for v in [0.0f32, -1.5, 0.831_204_7, 3.402_823_5e38, 1e-30] {
            let s = f32_json(v);
            assert_eq!(s.parse::<f32>().unwrap().to_bits(), v.to_bits(), "{s}");
        }
    }

    #[test]
    fn io_error_mapping() {
        let (c, _) = io_error(
            "read",
            "x",
            &std::io::Error::from(std::io::ErrorKind::NotFound),
        );
        assert_eq!(c, ERR_NOT_FOUND);
        let (c, _) = io_error(
            "read",
            "x",
            &std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        );
        assert_eq!(c, ERR_ACCESS_DENIED);
    }
}
