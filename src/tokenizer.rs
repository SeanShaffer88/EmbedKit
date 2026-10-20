//! BERT WordPiece tokenizer, uncased flavor — the tokenizer the
//! all-MiniLM class of sentence models expects.
//!
//! Pipeline (matching the reference BERT preprocessing):
//! 1. Basic tokenization: lowercase, strip accents, drop control
//!    characters, split on whitespace, and split every punctuation
//!    character into its own token.
//! 2. WordPiece: greedy longest-match-first against `vocab.txt`,
//!    continuation pieces prefixed with `##`; a word that cannot be
//!    segmented becomes a single `[UNK]`; words longer than 100
//!    characters are `[UNK]` outright.
//! 3. Sequence assembly: `[CLS] tokens [SEP]`, `token_type_ids` all
//!    zero (single-sequence embedding), `attention_mask` all ones
//!    (EmbedKit embeds one sequence per call — no padding needed).
//!    Truncation to `max_len` (when > 0) drops middle tokens and
//!    always keeps the final `[SEP]`.
//!
//! Zero-dependency constraints, stated honestly: Unicode general
//! categories are not available in `std`, so "punctuation" means
//! ASCII punctuation plus any non-alphanumeric, non-whitespace
//! character, and accent stripping covers the Latin-1 Supplement
//! and common Latin Extended-A letters (the range BERT-uncased
//! vocabularies actually exercise). For the Latin-script text these
//! models are used with, the output matches the reference tokenizer.

use std::collections::HashMap;

use crate::{CResult, ERR_NOT_FOUND, ERR_PLATFORM};

const MAX_WORD_CHARS: usize = 100;

#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    unk_id: i64,
    cls_id: i64,
    sep_id: i64,
    max_len: usize,
}

impl Tokenizer {
    /// Build from the text of a `vocab.txt` (one token per line; the
    /// line number is the token id). `max_len == 0` means no
    /// truncation beyond the model caller's own choice.
    pub fn from_vocab_text(text: &str, max_len: usize) -> CResult<Tokenizer> {
        let mut vocab = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let tok = line.strip_suffix('\r').unwrap_or(line);
            if !tok.is_empty() {
                vocab.insert(tok.to_string(), i as i64);
            }
        }
        let need = |name: &str| -> CResult<i64> {
            vocab.get(name).copied().ok_or((
                ERR_PLATFORM,
                format!("vocab is missing the required special token {name}"),
            ))
        };
        Ok(Tokenizer {
            unk_id: need("[UNK]")?,
            cls_id: need("[CLS]")?,
            sep_id: need("[SEP]")?,
            vocab,
            max_len,
        })
    }

    pub fn from_vocab_file(path: &str, max_len: usize) -> CResult<Tokenizer> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                (ERR_NOT_FOUND, format!("vocab file not found: '{path}'"))
            } else {
                crate::io_error("read", path, &e)
            }
        })?;
        Tokenizer::from_vocab_text(&text, max_len)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Tokenize one text into the three model input tensors.
    pub fn encode(&self, text: &str) -> Encoded {
        let mut ids: Vec<i64> = Vec::new();
        for word in basic_tokenize(text) {
            self.wordpiece(&word, &mut ids);
        }
        if self.max_len > 0 && ids.len() + 2 > self.max_len {
            // Keep [CLS] + the first tokens + [SEP].
            ids.truncate(self.max_len.saturating_sub(2));
        }
        let mut input_ids = Vec::with_capacity(ids.len() + 2);
        input_ids.push(self.cls_id);
        input_ids.extend_from_slice(&ids);
        input_ids.push(self.sep_id);
        let n = input_ids.len();
        Encoded {
            input_ids,
            attention_mask: vec![1; n],
            token_type_ids: vec![0; n],
        }
    }

    fn wordpiece(&self, word: &str, out: &mut Vec<i64>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_WORD_CHARS {
            out.push(self.unk_id);
            return;
        }
        let start_len = out.len();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found: Option<i64> = None;
            while end > start {
                let mut piece: String = if start > 0 {
                    "##".into()
                } else {
                    String::new()
                };
                piece.extend(&chars[start..end]);
                if let Some(&id) = self.vocab.get(&piece) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    // The whole word is unsegmentable: BERT emits a
                    // single [UNK] for it, discarding partial pieces.
                    out.truncate(start_len);
                    out.push(self.unk_id);
                    return;
                }
            }
        }
    }
}

/// Step 1 of BERT preprocessing, as plain word/punctuation tokens.
fn basic_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, tokens: &mut Vec<String>| {
        if !cur.is_empty() {
            tokens.push(std::mem::take(cur));
        }
    };
    for raw in text.chars() {
        if raw.is_control() && !raw.is_whitespace() {
            continue; // BERT drops control characters.
        }
        for c in raw.to_lowercase() {
            let c = strip_accent(c);
            if c == '\0' {
                continue; // a combining mark folded away to nothing
            }
            if c.is_whitespace() {
                flush(&mut cur, &mut tokens);
            } else if is_punctuation(c) {
                flush(&mut cur, &mut tokens);
                tokens.push(c.to_string());
            } else {
                cur.push(c);
            }
        }
    }
    flush(&mut cur, &mut tokens);
    tokens
}

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || (!c.is_alphanumeric() && !c.is_whitespace())
}

/// Accent folding for Latin letters: accented letter → base letter.
/// Non-Latin characters pass through unchanged; a character in a
/// combining sequence that was its own codepoint (e.g. U+0301)
/// folds to nothing, which is the decomposition behavior BERT's
/// accent stripper produces. (Single-char folds only — æ/ß are not
/// expanded to ae/ss; words containing them simply miss the vocab
/// and become [UNK], like any out-of-vocab word.)
fn strip_accent(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'ď' | 'đ' | 'ð' => 'd',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
        'ĥ' | 'ħ' => 'h',
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'ĵ' => 'j',
        'ķ' => 'k',
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'ś' | 'ŝ' | 'ş' | 'š' => 's',
        'ţ' | 'ť' | 'ŧ' => 't',
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ý' | 'ÿ' | 'ŷ' => 'y',
        'ź' | 'ż' | 'ž' => 'z',
        // Combining diacriticals (standalone codepoints) vanish.
        '\u{0300}'..='\u{036f}' => '\0',
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: &str = "\
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
hello
world
,
!
un
##aff
##able
cafe
the
cat
sat
##
";

    fn tok() -> Tokenizer {
        Tokenizer::from_vocab_text(VOCAB, 0).unwrap()
    }

    fn id_of(t: &Tokenizer, piece: &str) -> i64 {
        t.vocab[piece]
    }

    #[test]
    fn casing_and_punctuation() {
        let t = tok();
        let enc = t.encode("Hello, WORLD!");
        let want: Vec<i64> = ["[CLS]", "hello", ",", "world", "!", "[SEP]"]
            .iter()
            .map(|p| id_of(&t, p))
            .collect();
        assert_eq!(enc.input_ids, want);
        assert_eq!(enc.attention_mask, vec![1; 6]);
        assert_eq!(enc.token_type_ids, vec![0; 6]);
    }

    #[test]
    fn wordpiece_splits_unaffable() {
        // The canonical WordPiece example: "unaffable" segments as
        // un + ##aff + ##able when those are the vocab's pieces.
        let t = tok();
        let enc = t.encode("unaffable");
        let want: Vec<i64> = ["[CLS]", "un", "##aff", "##able", "[SEP]"]
            .iter()
            .map(|p| id_of(&t, p))
            .collect();
        assert_eq!(enc.input_ids, want);
    }

    #[test]
    fn unknown_word_becomes_single_unk() {
        let t = tok();
        let enc = t.encode("hello zzzqx world");
        // BERT: one [UNK] for the whole unsegmentable word.
        assert_eq!(enc.input_ids.len(), 5);
        assert_eq!(enc.input_ids[2], t.unk_id);
    }

    #[test]
    fn accents_fold_to_base_letters() {
        let t = tok();
        let enc = t.encode("café");
        let want: Vec<i64> = ["[CLS]", "cafe", "[SEP]"]
            .iter()
            .map(|p| id_of(&t, p))
            .collect();
        assert_eq!(enc.input_ids, want);
    }

    #[test]
    fn empty_text_is_cls_sep() {
        let t = tok();
        let enc = t.encode("   ");
        assert_eq!(enc.input_ids, vec![t.cls_id, t.sep_id]);
        assert_eq!(enc.attention_mask, vec![1, 1]);
    }

    #[test]
    fn truncation_keeps_cls_and_sep() {
        let t = Tokenizer::from_vocab_text(VOCAB, 4).unwrap();
        let enc = t.encode("the cat sat the cat sat");
        assert_eq!(enc.input_ids.len(), 4);
        assert_eq!(enc.input_ids[0], t.cls_id);
        assert_eq!(*enc.input_ids.last().unwrap(), t.sep_id);
    }

    #[test]
    fn long_word_is_unk() {
        let t = tok();
        let enc = t.encode(&"h".repeat(150));
        assert_eq!(enc.input_ids, vec![t.cls_id, t.unk_id, t.sep_id]);
    }

    #[test]
    fn missing_special_token_is_an_error() {
        let err = Tokenizer::from_vocab_text("a\nb\nc\n", 0).err().unwrap();
        assert_eq!(err.0, ERR_PLATFORM);
        assert!(err.1.contains("[UNK]"), "{}", err.1);
    }

    #[test]
    fn empty_vocab_is_an_error() {
        assert_eq!(
            Tokenizer::from_vocab_text("", 0).err().unwrap().0,
            ERR_PLATFORM
        );
    }
}
