//! Layer 1: the vector index — pure Rust, always available.
//!
//! An index has a fixed dimension chosen at creation. Entries carry
//! a string id, an optional text (the passage the vector represents,
//! echoed in search results), and the vector itself. Search is an
//! exact brute-force cosine scan — honest at v0.1 scale: scores are
//! the true cosines, ordering is deterministic (score descending,
//! ties by id ascending), and there is no approximation to explain
//! away. ANN indexes are roadmap, not v0.1.
//!
//! Persistence is the EKIDX1 binary format (all integers little
//! endian, offsets from the start of file):
//!
//! ```text
//! 0   magic          6 bytes  "EKIDX1"
//! 6   version        u32      = 1
//! 10  dim            u32      >= 1
//! 14  entry_count    u32
//! 18  entries, each:
//!       id_len       u16      >= 1
//!       id           id_len bytes, UTF-8
//!       has_text     u8       0 or 1
//!       text_len     u32      (only when has_text = 1)
//!       text         text_len bytes, UTF-8 (only when has_text = 1)
//!       vector       dim * f32
//! ```
//!
//! Loading validates every field against the actual buffer before
//! trusting it; a truncated or corrupt file is an error, never a
//! partial index.

use std::collections::HashMap;

use crate::{json_escape, CResult, ERR_INVALID_ARG, ERR_NOT_FOUND, ERR_PLATFORM};

const MAGIC: &[u8; 6] = b"EKIDX1";
const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub id: String,
    pub text: Option<String>,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub text: Option<String>,
}

pub struct Index {
    dim: usize,
    entries: Vec<Entry>,
    by_id: HashMap<String, usize>,
}

/// Cosine similarity; a zero-norm vector scores exactly 0.0 against
/// everything (defined behavior — a zero vector has no direction).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

impl Index {
    pub fn new(dim: usize) -> CResult<Index> {
        if dim == 0 {
            return Err((ERR_INVALID_ARG, "index dimension must be at least 1".into()));
        }
        Ok(Index {
            dim,
            entries: Vec::new(),
            by_id: HashMap::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn check_vector(&self, vector: &[f32]) -> CResult<()> {
        if vector.len() != self.dim {
            return Err((
                ERR_INVALID_ARG,
                format!(
                    "vector has dimension {}, index dimension is {}",
                    vector.len(),
                    self.dim
                ),
            ));
        }
        if vector.iter().any(|v| !v.is_finite()) {
            return Err((
                ERR_INVALID_ARG,
                "vector contains NaN or infinite values".into(),
            ));
        }
        Ok(())
    }

    /// Add an entry; an existing id is replaced in place.
    pub fn add(&mut self, id: &str, text: Option<String>, vector: Vec<f32>) -> CResult<()> {
        if id.is_empty() {
            return Err((ERR_INVALID_ARG, "entry id must not be empty".into()));
        }
        self.check_vector(&vector)?;
        let entry = Entry {
            id: id.to_string(),
            text,
            vector,
        };
        match self.by_id.get(id) {
            Some(&pos) => self.entries[pos] = entry,
            None => {
                self.by_id.insert(id.to_string(), self.entries.len());
                self.entries.push(entry);
            }
        }
        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> CResult<()> {
        match self.by_id.remove(id) {
            Some(pos) => {
                self.entries.remove(pos);
                // Positions after `pos` shifted; rebuild the map.
                self.by_id = self
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (e.id.clone(), i))
                    .collect();
                Ok(())
            }
            None => Err((ERR_NOT_FOUND, format!("no entry with id '{id}'"))),
        }
    }

    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.by_id.get(id).map(|&pos| &self.entries[pos])
    }

    /// Exact top-k cosine search. `k == 0` returns every entry,
    /// ranked. Ordering: score descending, id ascending on ties.
    pub fn search(&self, query: &[f32], k: usize) -> CResult<Vec<SearchHit>> {
        self.check_vector(query)?;
        let mut hits: Vec<SearchHit> = self
            .entries
            .iter()
            .map(|e| SearchHit {
                id: e.id.clone(),
                score: cosine(query, &e.vector),
                text: e.text.clone(),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        if k > 0 {
            hits.truncate(k);
        }
        Ok(hits)
    }

    pub fn save_to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            out.extend_from_slice(&(e.id.len() as u16).to_le_bytes());
            out.extend_from_slice(e.id.as_bytes());
            match &e.text {
                Some(t) => {
                    out.push(1);
                    out.extend_from_slice(&(t.len() as u32).to_le_bytes());
                    out.extend_from_slice(t.as_bytes());
                }
                None => out.push(0),
            }
            for v in &e.vector {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn load_from_bytes(bytes: &[u8]) -> CResult<Index> {
        let mut r = Reader { buf: bytes, pos: 0 };
        let corrupt = |why: &str| (ERR_PLATFORM, format!("corrupt EKIDX1 file: {why}"));
        if r.take(6)? != MAGIC {
            return Err(corrupt("bad magic"));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(corrupt(&format!("unsupported version {version}")));
        }
        let dim = r.u32()? as usize;
        if dim == 0 {
            return Err(corrupt("dimension is 0"));
        }
        let count = r.u32()? as usize;
        // Sanity bound: each entry needs at least 3 + dim*4 bytes, so
        // an absurd count is corrupt rather than a huge allocation.
        let min_entry = 3usize.saturating_add(dim.saturating_mul(4));
        if count > r.remaining() / min_entry.max(1) + 1 {
            return Err(corrupt("entry count exceeds file size"));
        }
        let mut index = Index::new(dim)?;
        for _ in 0..count {
            let id_len = r.u16()? as usize;
            if id_len == 0 {
                return Err(corrupt("empty id"));
            }
            let id = r.utf8(id_len)?;
            let text = match r.u8()? {
                0 => None,
                1 => {
                    let n = r.u32()? as usize;
                    Some(r.utf8(n)?)
                }
                other => return Err(corrupt(&format!("bad has_text flag {other}"))),
            };
            let mut vector = Vec::with_capacity(dim);
            for _ in 0..dim {
                vector.push(r.f32()?);
            }
            index
                .add(&id, text, vector)
                .map_err(|_| corrupt("duplicate or invalid entry"))?;
        }
        if r.remaining() != 0 {
            return Err(corrupt("trailing bytes after last entry"));
        }
        Ok(index)
    }
}

/// Bounds-checked little-endian cursor over the save bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> CResult<&'a [u8]> {
        if self.remaining() < n {
            return Err((
                ERR_PLATFORM,
                "corrupt EKIDX1 file: unexpected end of file".into(),
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> CResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> CResult<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> CResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> CResult<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn utf8(&mut self, n: usize) -> CResult<String> {
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| {
            (
                ERR_PLATFORM,
                "corrupt EKIDX1 file: invalid UTF-8".to_string(),
            )
        })
    }
}

/// Render search hits as the JSON array `ek_search` returns:
/// `[{"id": "doc1", "score": 0.831, "text": "..."}]` — the `text`
/// key is omitted for entries stored without text.
pub fn hits_json(hits: &[SearchHit]) -> String {
    let mut out = String::from("[");
    for (i, h) in hits.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str("{\"id\": \"");
        out.push_str(&json_escape(&h.id));
        out.push_str("\", \"score\": ");
        out.push_str(&crate::f32_json(h.score));
        if let Some(t) = &h.text {
            out.push_str(", \"text\": \"");
            out.push_str(&json_escape(t));
            out.push('"');
        }
        out.push('}');
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index3() -> Index {
        let mut ix = Index::new(2).unwrap();
        // Hand-computed cosines against the query [1, 0]:
        //   east  [1, 0]    -> 1.0
        //   diag  [0.8, .6] -> 0.8 (norm is exactly 1)
        //   north [0, 1]    -> 0.0
        ix.add("east", Some("pointing east".into()), vec![1.0, 0.0])
            .unwrap();
        ix.add("diag", None, vec![0.8, 0.6]).unwrap();
        ix.add("north", Some("pointing north".into()), vec![0.0, 1.0])
            .unwrap();
        ix
    }

    #[test]
    fn cosine_fixtures() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-7);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-7);
        assert!((cosine(&[1.0, 1.0], &[1.0, 0.0]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((cosine(&[2.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-7);
        // Zero vector: defined as 0, never NaN.
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn search_ordering_and_scores() {
        let ix = index3();
        let hits = ix.search(&[1.0, 0.0], 0).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, "east");
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert_eq!(hits[1].id, "diag");
        assert!((hits[1].score - 0.8).abs() < 1e-6);
        assert_eq!(hits[2].id, "north");
        assert!((hits[2].score - 0.0).abs() < 1e-6);
        // k truncates after ranking.
        let top1 = ix.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].id, "east");
        // Text rides along only where stored.
        assert_eq!(hits[0].text.as_deref(), Some("pointing east"));
        assert_eq!(hits[1].text, None);
    }

    #[test]
    fn add_replaces_and_remove_works() {
        let mut ix = index3();
        ix.add("diag", None, vec![0.0, 1.0]).unwrap();
        assert_eq!(ix.len(), 3);
        // diag now ties north at 0.0; tie-break is id ascending.
        let hits = ix.search(&[1.0, 0.0], 0).unwrap();
        assert_eq!(hits[1].id, "diag");
        assert_eq!(hits[2].id, "north");
        ix.remove("diag").unwrap();
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.remove("diag").err().unwrap().0, ERR_NOT_FOUND);
    }

    #[test]
    fn dim_validation() {
        let mut ix = Index::new(3).unwrap();
        assert_eq!(
            ix.add("x", None, vec![1.0, 2.0]).err().unwrap().0,
            ERR_INVALID_ARG
        );
        assert_eq!(ix.search(&[1.0], 1).err().unwrap().0, ERR_INVALID_ARG);
        assert_eq!(
            ix.add("x", None, vec![1.0, f32::NAN, 0.0]).err().unwrap().0,
            ERR_INVALID_ARG
        );
        assert_eq!(Index::new(0).err().unwrap().0, ERR_INVALID_ARG);
    }

    #[test]
    fn save_load_roundtrip() {
        let ix = index3();
        let bytes = ix.save_to_bytes();
        let back = Index::load_from_bytes(&bytes).unwrap();
        assert_eq!(back.dim(), 2);
        assert_eq!(back.len(), 3);
        let a = ix.search(&[0.3, 0.9], 0).unwrap();
        let b = back.search(&[0.3, 0.9], 0).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            back.get("east").unwrap().text.as_deref(),
            Some("pointing east")
        );
        assert_eq!(back.get("diag").unwrap().text, None);
    }

    #[test]
    fn load_rejects_corrupt_files() {
        let good = index3().save_to_bytes();
        // Bad magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(Index::load_from_bytes(&bad).err().unwrap().0, ERR_PLATFORM);
        // Truncated mid-entry.
        assert_eq!(
            Index::load_from_bytes(&good[..good.len() - 3])
                .err()
                .unwrap()
                .0,
            ERR_PLATFORM
        );
        // Trailing garbage.
        let mut bad = good.clone();
        bad.push(0);
        assert_eq!(Index::load_from_bytes(&bad).err().unwrap().0, ERR_PLATFORM);
        // Empty.
        assert_eq!(Index::load_from_bytes(&[]).err().unwrap().0, ERR_PLATFORM);
        // Header claims an absurd count.
        let mut bad = good.clone();
        bad[14..18].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(Index::load_from_bytes(&bad).err().unwrap().0, ERR_PLATFORM);
    }

    #[test]
    fn hits_json_shape() {
        let ix = index3();
        let j = hits_json(&ix.search(&[1.0, 0.0], 2).unwrap());
        assert!(
            j.starts_with("[{\"id\": \"east\", \"score\": 1, \"text\": \"pointing east\"}"),
            "{j}"
        );
        assert!(j.contains("{\"id\": \"diag\", \"score\": 0.8}"), "{j}");
    }
}
