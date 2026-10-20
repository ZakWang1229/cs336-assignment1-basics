use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

const MAX_VOCAB_SIZE: usize = 65535;
const MINIMAL_MERGE_CNT: usize = 1; // if for large train text, we don't care the unfrequent pairs
const PRETOKENIZE_PAT: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

type Pair = (u16, u16);

struct Chain {
    pre: Vec<usize>,
    nxt: Vec<usize>,
}
impl Chain {
    fn new(size: usize) -> Self {
        Self {
            pre: vec![size; size],
            nxt: vec![size; size],
        }
    }

    fn erase(&mut self, pos: usize) {
        let end = self.pre.len();
        let left = self.pre[pos];
        let right = self.nxt[pos];
        if left != end {
            self.nxt[left] = right;
        }
        if right != end {
            self.pre[right] = left;
        }
        self.pre[pos] = end;
        self.nxt[pos] = end;
    }
}

// Every pair occurrence is identified by its right token's buffer position.
struct PairIndex {
    len: usize,
    chain: Chain,
    counts: HashMap<Pair, (usize, usize)>, // count, occurrence-list head
    tails: HashMap<Pair, usize>,
    candidates: HashSet<Pair>,             // stable identities, not mutable head positions
}
impl PairIndex {
    fn new(size: usize) -> Self {
        Self {
            len: size,
            chain: Chain::new(size),
            counts: HashMap::new(),
            tails: HashMap::new(),
            candidates: HashSet::new(),
        }
    }

    fn insert(&mut self, pair: Pair, pos: usize) {
        let entry = self.counts.entry(pair).or_insert((0, self.len));
        let tail = self.tails.entry(pair).or_insert(self.len);
        self.chain.pre[pos] = *tail;
        self.chain.nxt[pos] = self.len;
        if *tail != self.len {
            debug_assert!(*tail < pos, "pair occurrences must be appended in text order");
            self.chain.nxt[*tail] = pos;
        } else {
            entry.1 = pos;
        }
        *tail = pos;
        entry.0 += 1;
        if entry.0 >= MINIMAL_MERGE_CNT { self.candidates.insert(pair); }
    }

    fn erase(&mut self, pair: Pair, pos: usize) {
        let entry = self
            .counts
            .get_mut(&pair)
            .expect("pair occurrence must exist");
        if entry.1 == pos { entry.1 = self.chain.nxt[pos]; }
        if self.tails[&pair] == pos {
            self.tails.insert(pair, self.chain.pre[pos]);
        }
        entry.0 -= 1;
        if entry.0 < MINIMAL_MERGE_CNT || entry.0 == 0 { self.candidates.remove(&pair); }
        self.chain.erase(pos);
        if entry.0 == 0 {
            self.counts.remove(&pair);
            self.tails.remove(&pair);
        }
    }
}

/// Train a byte-level BPE vocabulary and return Python bytes objects.
#[pyfunction]
#[pyo3(signature = (input_path, vocab_size, special_tokens, **kwargs))]
fn train_bpe(
    py: Python<'_>,
    input_path: PathBuf,
    vocab_size: usize,
    special_tokens: Vec<String>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let _ = kwargs;
    let mut specials = Vec::new();
    for token in special_tokens {
        if token.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "special tokens must not be empty",
            ));
        }
        if !specials.contains(&token) {
            specials.push(token);
        }
    }
    if vocab_size < 256 + specials.len() || vocab_size > MAX_VOCAB_SIZE {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid vocabulary size",
        ));
    }
    let text = std::fs::read_to_string(input_path)
        .map_err(|error| pyo3::exceptions::PyIOError::new_err(error.to_string()))?;
    let mut buff = pretokenlize(&text, specials.clone());
    let end = buff.len();
    let mut pos_chain = Chain::new(end);
    let mut pair_index = PairIndex::new(end);
    let mut vocab_map: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
    let mut vocab_merge = Vec::<(Vec<u8>, Vec<u8>)>::new();
    for pos in 1..end {
        if buff[pos - 1] != MAX_VOCAB_SIZE as u16 && buff[pos] != MAX_VOCAB_SIZE as u16 {
            pos_chain.pre[pos] = pos - 1;
            pos_chain.nxt[pos - 1] = pos;
            pair_index.insert((buff[pos - 1], buff[pos]), pos);
        }
    }
    while vocab_map.len() + specials.len() < vocab_size {
        let selected = pair_index.candidates.iter().copied().max_by(|a, b| {
            pair_index.counts[a]
                .0
                .cmp(&pair_index.counts[b].0)
                .then_with(|| vocab_map[a.0 as usize].cmp(&vocab_map[b.0 as usize]))
                .then_with(|| vocab_map[a.1 as usize].cmp(&vocab_map[b.1 as usize]))
        });
        let Some(selected) = selected else { break; };
        let left = vocab_map[selected.0 as usize].clone();
        let right = vocab_map[selected.1 as usize].clone();
        let token_id = vocab_map.len() as u16;
        let mut merged = Vec::<u8>::with_capacity(left.len() + right.len());
        merged.extend_from_slice(&left);
        merged.extend_from_slice(&right);
        vocab_map.push(merged);
        vocab_merge.push((left, right));

        let mut positions = Vec::with_capacity(pair_index.counts[&selected].0);
        let mut pos = pair_index.counts[&selected].1;
        while pos != end {
            positions.push(pos);
            pos = pair_index.chain.nxt[pos];
        }
        debug_assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        for pos2 in positions {
            let pos1 = pos_chain.pre[pos2];
            if pos1 == end || (buff[pos1], buff[pos2]) != selected { continue; }
            let pre_pos = pos_chain.pre[pos1];
            let nxt_pos = pos_chain.nxt[pos2];

            if pre_pos != end { pair_index.erase((buff[pre_pos], buff[pos1]), pos1); }
            pair_index.erase(selected, pos2);
            if nxt_pos != end { pair_index.erase((buff[pos2], buff[nxt_pos]), nxt_pos); }

            buff[pos2] = token_id;
            pos_chain.erase(pos1);
            if pre_pos != end { pair_index.insert((buff[pre_pos], token_id), pos2); }
            if nxt_pos != end { pair_index.insert((token_id, buff[nxt_pos]), nxt_pos); }
        }
    }
    vocab_map.extend(specials.into_iter().map(String::into_bytes));

    let vocab_py = PyDict::new(py);
    for (token_id, token_bytes) in vocab_map.iter().enumerate() {
        vocab_py.set_item(token_id, PyBytes::new(py, token_bytes))?;
    }
    let merges_py = PyList::empty(py);
    for (left, right) in &vocab_merge {
        merges_py.append((PyBytes::new(py, left), PyBytes::new(py, right)))?;
    }
    let result = PyTuple::new(py, [vocab_py.into_any(), merges_py.into_any()])?;
    Ok(result.into_any().unbind())
}

fn pretokenlize(s: &str, special_tokens: Vec<String>) -> Vec<u16> {
    let mut pos = 0usize;
    let mut text_vec = Vec::<&str>::new();
    while pos < s.len() {
        let mut end = s.len();
        let mut pos1 = s.len();
        let s_from_pos = &s[pos..];
        for pat in special_tokens.iter().map(String::as_str) {
            if let Some(x) = s_from_pos.find(pat) {
                let x_in_s = pos + x;
                if x_in_s < end || (x_in_s == end && x_in_s + pat.len() < pos1) {
                    end = x_in_s;
                    pos1 = x_in_s + pat.len();
                }
            }
        }
        text_vec.push(&s[pos..end]);
        pos = pos1;
    }

    let re = Regex::new(PRETOKENIZE_PAT).expect("pattern should compile");
    let mut buff = vec![MAX_VOCAB_SIZE as u16];
    for text in text_vec {
        for res in re.find_iter(text) {
            let slice = res.expect("matching should succeed").as_str().as_bytes();
            buff.extend(slice.iter().map(|v| *v as u16));
            buff.push(MAX_VOCAB_SIZE as u16);
        }
    }

    return buff;
}

/// Python extension entry point. Register your Rust functions here.
#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(train_bpe, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenization_preserves_text_and_expected_boundaries() {
        let text = "I'm learning 123!  中文";
        let re = Regex::new(PRETOKENIZE_PAT).expect("pattern should compile");
        let pieces: Vec<&str> = re
            .find_iter(text)
            .map(|result| result.expect("matching should succeed").as_str())
            .collect();

        assert_eq!(
            pieces,
            vec!["I", "'m", " learning", " 123", "!", " ", " 中文"]
        );
        assert_eq!(pieces.concat(), text);
    }
    #[test]
    fn pair_index_updates_head_and_unlinks_middle_and_tail() {
        let pair = (1, 2);
        let mut index = PairIndex::new(10);
        for pos in [1, 3, 5] {
            index.insert(pair, pos);
        }
        index.erase(pair, 3);
        assert_eq!(index.counts[&pair], (2, 1));
        assert_eq!(index.chain.nxt[1], 5);
        assert_eq!(index.chain.pre[5], 1);
        assert_eq!(index.tails[&pair], 5);
        index.erase(pair, 5);
        assert_eq!(index.tails[&pair], 1);
        assert_eq!(index.counts[&pair], (1, 1));
        assert_eq!(index.chain.pre[1], 10);
        index.erase(pair, 1);
        assert!(!index.counts.contains_key(&pair));
        assert!(!index.candidates.contains(&pair));
        index.insert((3, 4), 1);
        assert_eq!(index.chain.pre[1], 10);
        assert_eq!(index.chain.nxt[1], 10);
    }
}
