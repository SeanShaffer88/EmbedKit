# EmbedKit roadmap

v0.1 is an exact-search index + a BERT-class embedder over a
runtime-loaded ONNX Runtime. Where it goes next, in rough order:

## Search scale
- Approximate nearest-neighbor indexes (HNSW first) behind the
  same `ek_search` contract, with exact search kept as the
  reference implementation the ANN results are validated against.
- Memory-mapped EKIDX loading for large collections, and an
  EKIDX v2 with per-entry metadata (arbitrary key/value) once a
  real consumer needs it.

## Models & tokenizers
- Cased-model support (no lowercase/accent folding) and
  SentencePiece / BPE tokenizers for the Llama/MPNet families.
- Batch embedding API (`ek_embed_batch`) for indexing throughput.
- Optional CLS pooling and no-pooling modes for models that are
  not sentence-transformers-shaped.
- Model self-description: read the pooling/tokenizer expectations
  from a `config_sentence_transformers.json`-style sidecar instead
  of the v0.1 conventions.

## Runtime
- Execution providers beyond CPU (DirectML on Windows first),
  selected at embedder creation.
- A C-side smoke tool that loads the DLL and the runtime and
  prints the API table version checks, for support bundles.

## Hardening
- Fuzz the EKIDX1 loader and the tokenizer (both parse untrusted
  input) with cargo-fuzz in CI.
- Long-run soak: millions of add/search cycles against a valgrind /
  ASAN build of the cdylib to prove the ORT object lifecycle is
  leak-tight.
