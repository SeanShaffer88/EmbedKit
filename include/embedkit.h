/*
 * EmbedKit — local semantic search behind one clean C ABI.
 *
 * Two layers:
 *   1. A pure-Rust vector index: fixed dimension, string ids,
 *      optional stored text, exact (brute-force) cosine top-k
 *      search, EKIDX1 save/load. No model needed; always available.
 *   2. A model embedder for BERT-class sentence models (the
 *      all-MiniLM shape): WordPiece tokenization, ONNX inference,
 *      mean pooling, L2 normalization. The embedder loads ONNX
 *      Runtime at runtime from a caller-supplied library path —
 *      EmbedKit never links it — so the DLL itself has no external
 *      dependency beyond the OS.
 *
 * Conventions (collection house standard):
 *   - String inputs are `const char*` UTF-8, NUL-terminated.
 *   - `char*` results are caller-owned; free them with
 *     ek_free_string(). Do not free them any other way.
 *   - Index and embedder handles are opaque uint64_t values; 0 is
 *     never a valid handle. Handles are thread-safe: calls on
 *     distinct handles proceed in parallel; calls on one handle
 *     serialize. An embedder handle runs its session under its own
 *     lock — it is NOT re-entrant from the same thread through
 *     another EmbedKit call.
 *   - Failures return an EK_ERR_* code (or NULL / 0 / -1) and set a
 *     thread-local last-error message: read ek_last_error_code()
 *     and ek_last_error() on the same thread, immediately.
 *   - Nothing unwinds or longjmps across this boundary.
 *
 * Error codes:
 *   EK_OK 0 · EK_ERR_INVALID_ARG 1 · EK_ERR_NOT_SUPPORTED 2 ·
 *   EK_ERR_NOT_FOUND 3 · EK_ERR_ACCESS_DENIED 4 ·
 *   EK_ERR_PLATFORM 5 · EK_ERR_INTERNAL 6
 */
#ifndef EMBEDKIT_H
#define EMBEDKIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EK_OK 0
#define EK_ERR_INVALID_ARG 1
#define EK_ERR_NOT_SUPPORTED 2
#define EK_ERR_NOT_FOUND 3
#define EK_ERR_ACCESS_DENIED 4
#define EK_ERR_PLATFORM 5
#define EK_ERR_INTERNAL 6

/* ---------------------------------------------------------------- */
/* Version & errors                                                  */
/* ---------------------------------------------------------------- */

/* Static "0.1.0" string; do not free. */
const char* ek_version(void);
uint32_t ek_version_major(void);
uint32_t ek_version_minor(void);
uint32_t ek_version_patch(void);

/* Thread-local last-error message (static storage; do not free). */
const char* ek_last_error(void);
int ek_last_error_code(void);

/* Free a string this library returned. NULL is a no-op. */
void ek_free_string(char* ptr);

/* ---------------------------------------------------------------- */
/* Index (layer 1)                                                   */
/* ---------------------------------------------------------------- */

/* Create an empty index of the given fixed dimension (>= 1).
 * Returns a handle, or 0 on failure. */
uint64_t ek_index_create(uint32_t dim);

/* Free an index. EK_OK, or EK_ERR_NOT_FOUND for a bad handle. */
int ek_index_free(uint64_t index);

/* Entry count, or -1 on error. */
int64_t ek_index_count(uint64_t index);

/* Add an entry (an existing id is replaced). `text` may be NULL;
 * stored text is echoed in search results. `vector` must point to
 * `dim` readable floats and `dim` must equal the index dimension;
 * NaN/infinite components are EK_ERR_INVALID_ARG. */
int ek_index_add(uint64_t index, const char* id, const char* text,
                 const float* vector, uint32_t dim);

/* Embed `text` with `embedder` and add it under `id`; the text is
 * stored as the entry's text. The embedder's dimension must equal
 * the index dimension. */
int ek_index_add_text(uint64_t index, uint64_t embedder, const char* id,
                      const char* text);

/* Remove an entry. EK_OK, or EK_ERR_NOT_FOUND. */
int ek_index_remove(uint64_t index, const char* id);

/* Exact cosine top-k search. Returns a caller-owned JSON array:
 *   [{"id": "doc1", "score": 0.831, "text": "the passage"}, ...]
 * The "text" key is omitted for entries stored without text.
 * `k <= 0` returns every entry, ranked. Scores are true cosine
 * similarities in [-1, 1]; ordering is score descending, ties by
 * id ascending. NULL on failure. */
char* ek_search(uint64_t index, const float* query, uint32_t dim, int k);

/* Embed `query_text` with `embedder` and search with the result.
 * Same JSON, same k semantics as ek_search. NULL on failure. */
char* ek_search_text(uint64_t index, uint64_t embedder,
                     const char* query_text, int k);

/* Save an index in the EKIDX1 format (below). */
int ek_index_save(uint64_t index, const char* path);

/* Load an EKIDX1 file as a new index; returns its handle, or 0 on
 * failure (missing file -> EK_ERR_NOT_FOUND; corrupt ->
 * EK_ERR_PLATFORM). */
uint64_t ek_index_load(const char* path);

/*
 * EKIDX1 file format (all integers little-endian):
 *   0   magic        6 bytes "EKIDX1"
 *   6   version      u32 = 1
 *   10  dim          u32 >= 1
 *   14  entry_count  u32
 *   18  entries: u16 id_len, id bytes (UTF-8),
 *                u8 has_text (0/1), [u32 text_len, text bytes],
 *                dim * f32 vector
 * Loading validates every length against the file size; truncated
 * or corrupt files are rejected whole, never loaded partially.
 */

/* ---------------------------------------------------------------- */
/* Embedder (layer 2)                                                */
/* ---------------------------------------------------------------- */

/* Create an embedder.
 *   ort_lib_path: path to the ONNX Runtime shared library
 *     (onnxruntime.dll / libonnxruntime.so). Loaded at runtime;
 *     EmbedKit v0.1 needs a runtime serving ONNX C API version 31
 *     (ONNX Runtime 1.31.0 is the pinned, verified one; newer
 *     runtimes remain compatible).
 *   model_path: a BERT-class sentence model in ONNX form whose
 *     inputs are a subset of input_ids / attention_mask /
 *     token_type_ids and whose token-embeddings output is named
 *     last_hidden_state (or is its only/first output).
 *   vocab_path: the model's WordPiece vocab.txt.
 *   max_len: token cap per text (including [CLS]/[SEP]);
 *     0 selects the default of 256.
 * Returns a handle, or 0 on failure. Missing files are
 * EK_ERR_NOT_FOUND (checked before any load attempt); a model of
 * the wrong shape is EK_ERR_NOT_SUPPORTED with the reason in
 * ek_last_error. */
uint64_t ek_embedder_create(const char* ort_lib_path,
                            const char* model_path,
                            const char* vocab_path, uint32_t max_len);

/* Free an embedder, releasing its session and runtime.
 * EK_OK, or EK_ERR_NOT_FOUND for a bad handle. */
int ek_embedder_free(uint64_t embedder);

/* The model's embedding dimension, or -1 on error. */
int64_t ek_embedder_dim(uint64_t embedder);

/* Embed one text. Returns a caller-owned JSON string:
 *   {"dim": 384, "vector": [0.0123, ...]}
 * Floats print in shortest round-trip form, so parsing them back
 * yields bit-identical f32 values. The vector is mean-pooled over
 * real tokens and L2-normalized. NULL on failure. */
char* ek_embed(uint64_t embedder, const char* text);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDKIT_H */
