# EmbedKit — agent notes

- Zero Rust dependencies is a design rule. The ONE external
  component is ONNX Runtime, and it is never linked and never
  committed: `src/ort.rs` loads the caller's library at runtime
  (the WinRecovery pattern) and calls the C API table from
  `OrtGetApiBase`. Do not add an `ort`-style binding crate or a
  windows-binding dependency without a written justification.
- The OrtApi field indices in src/ort.rs are ABI: they were
  transcribed from the pinned runtime's onnxruntime_c_api.h
  (ORT 1.31.0, ORT_API_VERSION 31) with a parser, and the order is
  load-bearing. Bumping the pinned runtime = bump pins.txt AND
  re-transcribe any changed indices AND re-run the e2e proof, in
  one commit. GetApi(31) failing means "runtime too old" — report
  NOT_SUPPORTED with its version string; never fall back to a
  guessed table layout.
- Two build flavors are both first-class: runtime-absent (fresh
  checkout — FFI compiles, all unit tests pass, ek_embedder_create
  reports precise NOT_FOUND/NOT_SUPPORTED) and runtime-present.
  The WinDevKit gate must stay green with NO fixtures downloaded;
  never make the host gate depend on the ~180 MB of fixtures.
- Embedder lifecycle is load-bearing: release order in
  Embedder::drop is session → memory info → env → library unload,
  and construction fills the struct progressively so Drop cleans
  up every failure path. Probe run at create ("" → [CLS] [SEP])
  is what fixes ek_embedder_dim; keep it.
- Claims discipline (SQLiteSuperBuild rule): the README's numbers
  are the e2e's measured cosines on the pinned model/runtime —
  nothing broader. The e2e asserts semantic structure (margins,
  ranking), never exact floats; keep it that way.
- Gate: WinDevKit tools/gate.sh (fmt + clippy BOTH targets + full
  tests). Check/clippy do not link — CI's real build plus the
  labeled Windows e2e step are the link + native-runtime gates.
  Windows-native runtime loading (LoadLibraryExW path) is
  UNVERIFIED until that CI step runs; do not claim otherwise.
