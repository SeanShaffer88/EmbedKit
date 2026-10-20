//! Layer 2: the model embedder.
//!
//! An embedder owns one ONNX Runtime session for one BERT-class
//! sentence model, its WordPiece tokenizer, and the loaded runtime
//! itself; dropping it releases the session, the memory info, the
//! env, and finally the library, in that order. Construction fills
//! the struct progressively with null ORT handles, so an error at
//! any step unwinds through `Drop` and releases whatever exists —
//! no partial-construction leaks.
//!
//! Creation validates everything up front and fails precisely:
//! missing files are NOT_FOUND (checked before any load attempt),
//! an unloadable or too-old runtime is the error `ort` reports, a
//! model whose inputs are not the BERT trio
//! (input_ids / attention_mask / token_type_ids — fed in whatever
//! subset and order the model declares) is NOT_SUPPORTED, and a
//! successful creation ends with a probe run on an empty sequence
//! that fixes the embedding dimension `ek_embedder_dim` reports.

use std::ffi::CString;
use std::os::raw::c_void;

use crate::ort::Ort;
use crate::tokenizer::Tokenizer;
use crate::{require_file, CResult, ERR_NOT_SUPPORTED, ERR_PLATFORM};

/// Which model input a tensor feeds.
#[derive(Clone, Copy, PartialEq)]
enum Feed {
    Ids,
    Mask,
    Types,
}

/// Default sequence cap when the caller passes max_len = 0: the
/// all-MiniLM family's sentence-transformers configuration value.
pub const DEFAULT_MAX_LEN: usize = 256;

pub struct Embedder {
    ort: Ort,
    env: *mut c_void,
    session: *mut c_void,
    mem_info: *mut c_void,
    inputs: Vec<(CString, Feed)>,
    output_name: CString,
    tokenizer: Tokenizer,
    dim: usize,
}

// The raw ORT objects are owned by exactly this struct and only ever
// touched while holding the per-embedder mutex in the FFI registry.
unsafe impl Send for Embedder {}

impl Embedder {
    pub fn create(
        ort_lib_path: &str,
        model_path: &str,
        vocab_path: &str,
        max_len: usize,
    ) -> CResult<Embedder> {
        require_file(ort_lib_path, "ONNX Runtime library")?;
        require_file(model_path, "model file")?;
        require_file(vocab_path, "vocab file")?;

        let eff_max_len = if max_len == 0 {
            DEFAULT_MAX_LEN
        } else {
            max_len.max(2)
        };
        let tokenizer = Tokenizer::from_vocab_file(vocab_path, eff_max_len)?;
        let ort = Ort::load(ort_lib_path)?;
        let env = ort.create_env()?;
        let mut emb = Embedder {
            ort,
            env,
            session: std::ptr::null_mut(),
            mem_info: std::ptr::null_mut(),
            inputs: Vec::new(),
            output_name: CString::default(),
            tokenizer,
            dim: 0,
        };
        // From here on, returning Err drops `emb`, whose Drop
        // releases the session / memory info / env created so far.
        // SAFETY: env is the live OrtEnv created just above, and
        // emb.session is the live session created just here; both
        // are owned by `emb` and released only in its Drop.
        emb.session = unsafe { emb.ort.create_session(env, model_path) }?;
        for name in unsafe { emb.ort.session_input_names(emb.session) }? {
            let kind = match name.as_str() {
                "input_ids" => Feed::Ids,
                "attention_mask" => Feed::Mask,
                "token_type_ids" => Feed::Types,
                other => {
                    return Err((
                        ERR_NOT_SUPPORTED,
                        format!(
                            "model input '{other}' is not one of input_ids / \
                             attention_mask / token_type_ids — not a BERT-class \
                             sentence model EmbedKit v0.1 supports"
                        ),
                    ))
                }
            };
            emb.inputs.push((CString::new(name).unwrap(), kind));
        }
        if !emb.inputs.iter().any(|(_, k)| *k == Feed::Ids) {
            return Err((
                ERR_NOT_SUPPORTED,
                "model has no input_ids input — not a BERT-class sentence model".into(),
            ));
        }
        let outputs = unsafe { emb.ort.session_output_names(emb.session) }?;
        if outputs.is_empty() {
            return Err((ERR_PLATFORM, "model declares no outputs".into()));
        }
        // The sentence-transformers exports name the token-embeddings
        // output "last_hidden_state"; otherwise the first output is
        // used and the probe below verifies it has the right shape.
        let output_name = if outputs.iter().any(|n| n == "last_hidden_state") {
            "last_hidden_state".to_string()
        } else {
            outputs[0].clone()
        };
        emb.output_name = CString::new(output_name).unwrap();
        emb.mem_info = emb.ort.create_cpu_memory_info()?;
        // Probe: an empty sequence ([CLS] [SEP]) fixes the hidden
        // size and proves the whole path before any handle escapes.
        let probe = emb.embed("")?;
        emb.dim = probe.len();
        Ok(emb)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn ort_version(&self) -> &str {
        &self.ort.version_string
    }

    /// Embed one text: tokenize, run the model, mean-pool the last
    /// hidden state over real tokens, L2-normalize.
    pub fn embed(&self, text: &str) -> CResult<Vec<f32>> {
        let enc = self.tokenizer.encode(text);
        let seq = enc.input_ids.len();
        let feeds: Vec<(CString, &[i64])> = self
            .inputs
            .iter()
            .map(|(name, kind)| {
                let data: &[i64] = match kind {
                    Feed::Ids => &enc.input_ids,
                    Feed::Mask => &enc.attention_mask,
                    Feed::Types => &enc.token_type_ids,
                };
                (name.clone(), data)
            })
            .collect();
        // SAFETY: session and mem_info are this embedder's live
        // ORT objects (created in `create`, released only in Drop),
        // and the feed buffers in `feeds` outlive the call.
        let out = unsafe {
            self.ort
                .run_int64_to_float(self.session, self.mem_info, &feeds, &self.output_name)
        }?;
        if out.dims.len() != 3 || out.dims[0] != 1 || out.dims[1] as usize != seq {
            return Err((
                ERR_PLATFORM,
                format!(
                    "model output shape {:?} is not [1, {seq}, hidden] — \
                     not a BERT-style encoder EmbedKit v0.1 supports",
                    out.dims
                ),
            ));
        }
        let dim = out.dims[2] as usize;
        if self.dim != 0 && dim != self.dim {
            return Err((
                ERR_PLATFORM,
                format!(
                    "model hidden size changed between runs ({dim} vs {})",
                    self.dim
                ),
            ));
        }
        Ok(mean_pool_normalized(
            &out.data,
            seq,
            dim,
            &enc.attention_mask,
        ))
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        // SAFETY: each pointer is the live object this embedder
        // created (or null during a failed construction, which the
        // release functions accept), released exactly once, here.
        unsafe {
            self.ort.release_session(self.session);
            self.ort.release_memory_info(self.mem_info);
            self.ort.release_env(self.env);
        }
        // The Ort (and with it the loaded library) drops after this,
        // when no ORT object can still reference it.
    }
}

/// Attention-mask-aware mean pooling + L2 normalization — the
/// sentence-transformers recipe for BERT token embeddings. `hidden`
/// is `[seq][dim]` row-major; masked-out positions contribute
/// nothing. A zero result (empty mask) stays zero, never NaN.
pub fn mean_pool_normalized(hidden: &[f32], seq: usize, dim: usize, mask: &[i64]) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let mut count = 0usize;
    for t in 0..seq {
        if mask.get(t).copied().unwrap_or(0) == 0 {
            continue;
        }
        count += 1;
        let row = &hidden[t * dim..(t + 1) * dim];
        for (o, &v) in out.iter_mut().zip(row.iter()) {
            *o += v;
        }
    }
    if count > 0 {
        let inv = 1.0 / count as f32;
        for o in &mut out {
            *o *= inv;
        }
    }
    l2_normalize(&mut out);
    out
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let inv = (1.0 / norm) as f32;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_ignores_masked_positions() {
        // Rows: [1,0] (real), [0,1] (real), [9,9] (padding).
        let hidden = [1.0, 0.0, 0.0, 1.0, 9.0, 9.0];
        let out = mean_pool_normalized(&hidden, 3, 2, &[1, 1, 0]);
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out[0] - s).abs() < 1e-6, "{out:?}");
        assert!((out[1] - s).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn pooling_single_token_is_that_token_normalized() {
        let out = mean_pool_normalized(&[3.0, 4.0], 1, 2, &[1]);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn pooling_empty_mask_is_zero_not_nan() {
        let out = mean_pool_normalized(&[5.0, 5.0], 1, 2, &[0]);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn create_with_missing_files_is_not_found() {
        let err = Embedder::create("no/lib.so", "no/model.onnx", "no/vocab.txt", 0)
            .err()
            .unwrap();
        assert_eq!(err.0, crate::ERR_NOT_FOUND, "{err:?}");
        // Lib + model exist (the manifest stands in as an existing
        // file twice), vocab missing → still NOT_FOUND, reported
        // before any load attempt.
        let here = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let err = Embedder::create(here, here, "no/vocab.txt", 0)
            .err()
            .unwrap();
        assert_eq!(err.0, crate::ERR_NOT_FOUND);
        assert!(err.1.contains("vocab"), "{}", err.1);
    }
}
