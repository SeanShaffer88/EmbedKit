//! End-to-end embedding proof — the real ONNX Runtime, the real
//! all-MiniLM-L6-v2 model, real sentences. Ignored by default; run
//! explicitly (after fetch.sh / fetch.ps1):
//!
//!   EK_E2E=1 \
//!   EK_ORT_LIB=fixtures/ort/lib/libonnxruntime.so.1.31.0 \
//!   EK_MODEL=fixtures/model.onnx EK_VOCAB=fixtures/vocab.txt \
//!   cargo test --test e2e -- --ignored --nocapture
//!
//! The assertions are SEMANTIC, not numeric: a paraphrase must sit
//! much closer to the anchor sentence than unrelated sentences do,
//! and a text search must rank the anchor's paraphrase first. This
//! is the proof that the tokenizer, the runtime binding, the
//! pooling, and the index all work together — exact-float equality
//! across runtimes would prove nothing and break spuriously.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

use embedkit::ffi::{
    ek_embed, ek_embedder_create, ek_embedder_dim, ek_embedder_free, ek_free_string, ek_index_add,
    ek_index_add_text, ek_index_count, ek_index_create, ek_index_free, ek_index_load,
    ek_index_save, ek_last_error, ek_search, ek_search_text,
};
use embedkit::index::cosine;

fn env_path(name: &str) -> String {
    let v =
        std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set to run the e2e test"));
    assert!(
        Path::new(&v).is_file(),
        "{name}={v} is not a file (run fetch.sh / fetch.ps1 first)"
    );
    v
}

fn cs(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn take(p: *mut c_char) -> String {
    assert!(
        !p.is_null(),
        "call returned null; last error: {}",
        unsafe { CStr::from_ptr(ek_last_error()) }.to_string_lossy()
    );
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
    ek_free_string(p);
    s
}

/// Minimal extraction of the "vector" array from ek_embed's JSON —
/// the shape is ours and fixed, so a full JSON parser is overkill.
fn parse_vector(json: &str) -> Vec<f32> {
    let start = json.find("\"vector\": [").expect("vector key") + "\"vector\": [".len();
    let end = json.rfind(']').expect("closing bracket");
    json[start..end]
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("float"))
        .collect()
}

const ANCHOR: &str = "A man is eating food.";
const PARAPHRASE: &str = "A man is eating a meal.";
const FAR: [&str; 3] = [
    "A woman is riding a horse.",
    "The stock market fell sharply today.",
    "A dog is sleeping on the porch.",
];

#[test]
#[ignore = "needs ONNX Runtime + pinned model/vocab (EK_E2E, EK_ORT_LIB, EK_MODEL, EK_VOCAB)"]
fn semantic_embedding_proof() {
    assert_eq!(
        std::env::var("EK_E2E").as_deref(),
        Ok("1"),
        "set EK_E2E=1 to run the e2e embedding proof"
    );
    let ort = env_path("EK_ORT_LIB");
    let model = env_path("EK_MODEL");
    let vocab = env_path("EK_VOCAB");

    let emb = ek_embedder_create(
        cs(&ort).as_ptr(),
        cs(&model).as_ptr(),
        cs(&vocab).as_ptr(),
        0,
    );
    assert!(emb != 0, "embedder creation failed");
    let dim = ek_embedder_dim(emb);
    println!("embedder dim: {dim}");
    assert_eq!(dim, 384, "all-MiniLM-L6-v2 embeds at 384");

    let embed = |text: &str| parse_vector(&take(ek_embed(emb, cs(text).as_ptr())));
    let anchor = embed(ANCHOR);
    let para = embed(PARAPHRASE);
    assert_eq!(anchor.len(), 384);
    let cos_para = cosine(&anchor, &para);
    println!("cos(anchor, paraphrase) = {cos_para:.4}");
    assert!(
        cos_para > 0.5,
        "paraphrase cosine suspiciously low: {cos_para}"
    );
    for far_text in FAR {
        let far = embed(far_text);
        let cos_far = cosine(&anchor, &far);
        println!("cos(anchor, {far_text:?}) = {cos_far:.4}");
        assert!(
            cos_para > cos_far + 0.1,
            "semantic margin too small: paraphrase {cos_para:.4} vs {far_text:?} {cos_far:.4}"
        );
    }

    // Full text-search flow through the index.
    let ix = ek_index_create(384);
    assert!(ix != 0);
    let sentences: [&str; 5] = [ANCHOR, PARAPHRASE, FAR[0], FAR[1], FAR[2]];
    for (i, s) in sentences.iter().enumerate() {
        let rc = ek_index_add_text(ix, emb, cs(&format!("s{i}")).as_ptr(), cs(s).as_ptr());
        assert_eq!(rc, embedkit::EK_OK, "add_text s{i}");
    }
    assert_eq!(ek_index_count(ix), 5);

    // Query with the anchor: s0 (itself) first, s1 (paraphrase) next.
    let j = take(ek_search_text(ix, emb, cs(ANCHOR).as_ptr(), 0));
    println!("search(anchor): {j}");
    let pos = |id: &str| j.find(&format!("\"id\": \"{id}\"")).expect(id);
    assert!(pos("s0") < pos("s1"), "{j}");
    assert!(pos("s1") < pos("s2"), "{j}");
    assert!(pos("s1") < pos("s3"), "{j}");
    assert!(pos("s1") < pos("s4"), "{j}");

    // Raw-vector search agrees: the anchor vector finds s0 first.
    let j2 = take(ek_search(ix, anchor.as_ptr(), 384, 1));
    assert!(j2.contains("\"id\": \"s0\""), "{j2}");

    // Persistence round-trip through EKIDX1 keeps the ranking.
    let path = std::env::temp_dir().join(format!("ek-e2e-{}.ekidx", std::process::id()));
    let p = cs(path.to_str().unwrap());
    assert_eq!(ek_index_save(ix, p.as_ptr()), embedkit::EK_OK);
    let ix2 = ek_index_load(p.as_ptr());
    assert!(ix2 != 0);
    assert_eq!(ek_index_count(ix2), 5);
    let j3 = take(ek_search(ix2, anchor.as_ptr(), 384, 2));
    assert!(j3.contains("\"id\": \"s0\""), "{j3}");
    assert!(j3.contains("\"id\": \"s1\""), "{j3}");
    let _ = std::fs::remove_file(&path);

    assert_eq!(ek_index_free(ix), embedkit::EK_OK);
    assert_eq!(ek_index_free(ix2), embedkit::EK_OK);
    assert_eq!(ek_embedder_free(emb), embedkit::EK_OK);
    // The handle is gone now.
    assert_eq!(ek_embedder_dim(emb), -1);

    // Layer 1 is independent of layer 2: a raw index still works
    // after every embedder is gone.
    let raw = ek_index_create(2);
    let v = [1.0f32, 0.0];
    assert_eq!(
        ek_index_add(raw, cs("x").as_ptr(), std::ptr::null(), v.as_ptr(), 2),
        embedkit::EK_OK
    );
    assert_eq!(ek_index_free(raw), embedkit::EK_OK);
}
