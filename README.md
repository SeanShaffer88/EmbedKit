# EmbedKit

Local semantic search behind one clean C ABI — DLL #12 in Sean's
Windows DLL Collection. Two layers, both fully working:

1. **A vector index** (pure Rust, no dependencies, always
   available): store embedding vectors under string ids with their
   text, and run exact cosine-similarity top-k search over them.
   Save/load in a documented binary format (EKIDX1).
2. **A model embedder** for BERT-class sentence models (the
   all-MiniLM shape): WordPiece tokenization, ONNX inference,
   attention-mask mean pooling, L2 normalization — turning raw text
   into vectors the index can store and search.

Everything runs on the local machine. There is no network code in
the DLL, no telemetry, and no account.

## The ONNX Runtime story (read this first)

Layer 2 runs models with **ONNX Runtime** (Microsoft, MIT), but
EmbedKit **never links it and never bundles it**. At
`ek_embedder_create` you pass the path to your
`onnxruntime.dll` / `libonnxruntime.so`; EmbedKit loads it at
runtime and speaks its C API. The DLL you ship stays free of
external dependencies beyond the OS, and the runtime version is
*yours* to choose and update.

- EmbedKit v0.1 speaks ONNX Runtime **C API version 31**
  (ONNX Runtime 1.31.0). Newer runtimes remain compatible; an older
  runtime is refused with a precise `EK_ERR_NOT_SUPPORTED` message
  naming the version it reported.
- `pins.txt` pins the exact 1.31.0 packages (with SHA-256) as a
  known-good source; `fetch.ps1` / `fetch.sh` download and
  hash-verify them into `fixtures/` (gitignored) for development
  and for the end-to-end proof. `fetch` is a developer convenience
  — the DLL itself downloads nothing.

## Getting a model

Any BERT-uncased-class sentence model in ONNX form works if its
inputs are a subset of `input_ids` / `attention_mask` /
`token_type_ids` and it exposes token embeddings as
`last_hidden_state` (or as its first output). The reference model —
used by the e2e proof, pinned in `pins.txt` — is
**all-MiniLM-L6-v2** (384 dimensions) from its official
sentence-transformers repository on Hugging Face:
`onnx/model.onnx` + `vocab.txt`.

## Quick start (C)

```c
#include "embedkit.h"

// Layer 2: one embedder, loaded once.
uint64_t emb = ek_embedder_create(
    "onnxruntime.dll", "model.onnx", "vocab.txt", 0);
int64_t dim = ek_embedder_dim(emb);            // 384 for MiniLM

// Layer 1: one index of that dimension.
uint64_t ix = ek_index_create((uint32_t)dim);
ek_index_add_text(ix, emb, "doc1", "A man is eating food.");
ek_index_add_text(ix, emb, "doc2", "A man is eating a meal.");
ek_index_add_text(ix, emb, "doc3", "A woman is riding a horse.");

// Semantic search: doc2 outranks doc3 for this query.
char* hits = ek_search_text(ix, emb, "A man is eating food.", 2);
// [{"id": "doc1", "score": 1, "text": "A man is eating food."},
//  {"id": "doc2", "score": 0.926..., "text": "A man is eating a meal."}]
ek_free_string(hits);

ek_index_save(ix, "docs.ekidx");               // EKIDX1 format
ek_index_free(ix);
ek_embedder_free(emb);
```

Already have vectors (from any source)? Skip layer 2:
`ek_index_add` takes raw `float*` arrays, and `ek_search` takes a
raw query vector. Layer 1 never needs a model or a runtime.

## The ABI at a glance

Full documentation lives in `include/embedkit.h`. Summary:

- Version/errors: `ek_version*`, `ek_last_error*`, `ek_free_string`
- Index: `ek_index_create/free/count`, `ek_index_add`,
  `ek_index_add_text`, `ek_index_remove`, `ek_search`,
  `ek_search_text`, `ek_index_save/load`
- Embedder: `ek_embedder_create/free`, `ek_embedder_dim`,
  `ek_embed`

House conventions: caller-owned result strings (free with
`ek_free_string`), opaque u64 handles, thread-local last error,
error codes 0–6 (OK / INVALID_ARG / NOT_SUPPORTED / NOT_FOUND /
ACCESS_DENIED / PLATFORM / INTERNAL), nothing unwinds across the
boundary. Handles are thread-safe; calls on one handle serialize.

## What the tests prove

- **Unit** (28 tests, no downloads): WordPiece against hand-checked
  examples (the canonical `unaffable` split, punctuation, casing,
  accent folding, `[UNK]`, truncation), pooling math on fixed
  tensors, hand-computed cosine fixtures, index ordering /
  replace / remove, EKIDX1 round-trip and corrupt-file rejection,
  and the FFI error contract (bad handles, dim mismatches, missing
  files).
- **End-to-end** (`tests/e2e.rs`, env-gated, run by CI): the real
  ONNX Runtime 1.31.0 + the real all-MiniLM-L6-v2 model embed real
  sentences. Locally measured: `cos("A man is eating food.",
  "A man is eating a meal.") = 0.9262`, vs `-0.1174` for "A woman
  is riding a horse.", `0.0037` for "The stock market fell sharply
  today.", and `0.0050` for "A dog is sleeping on the porch." A
  five-sentence text search ranks the anchor first and its
  paraphrase second, and the ranking survives an EKIDX1
  save/load round-trip.

## v0.1 limits, honestly

- **Search is brute force.** Exact and deterministic — the right
  default at this scale — but O(n·dim) per query. ANN indexes are
  on the roadmap, not in v0.1.
- **BERT-uncased-class models only.** WordPiece with lowercasing
  and accent folding (Latin scripts; single-char folds — æ/ß are
  not expanded). Other tokenizer families (SentencePiece/BPE,
  cased models) are roadmap.
- **One sequence at a time.** No batching API; each `ek_embed`
  runs one text. Calls on one embedder handle serialize on its
  lock (ONNX sessions run under it) — use several embedder handles
  for parallel embedding.
- **CPU execution only.** GPU execution providers are roadmap.
- The embedder accepts a model whose token-embeddings output is
  its first output when it isn't named `last_hidden_state`, and
  verifies the `[1, seq, hidden]` shape on a probe run at
  creation; anything else is refused with the reason, up front.

## Building

```
cargo build --release        # produces embedkit.dll / libembedkit.so
```

The Rust side is zero-dependency. For the e2e proof:
`./fetch.ps1` (Windows) or `sh ./fetch.sh` (Linux), then the
command printed at the end of the fetch.

## Layout

- `src/index.rs` — layer 1: index, cosine, EKIDX1
- `src/tokenizer.rs` — WordPiece (BERT uncased)
- `src/ort.rs` — the runtime-loaded ONNX Runtime C API binding;
  its OrtApi field indices are transcribed from the pinned
  runtime's own header (see the module docs)
- `src/embedder.rs` — layer 2: session ownership, pooling
- `src/ffi.rs` — the C ABI
- `include/embedkit.h` — the ABI contract, including the EKIDX1
  byte layout

MIT — see `LICENSE`. ONNX Runtime is a separate component under
its own MIT license, supplied by the user at runtime.
