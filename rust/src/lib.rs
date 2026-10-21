use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

const MAX_VOCAB_SIZE: usize = 65535;
const PRETOKENIZE_PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

macro_rules! py_bail {
    ($exception:ident, $($message:tt)+) => {
        return Err(pyo3::exceptions::$exception::new_err(format!($($message)+)))
    };
}

type Pair = (u16, u16);

struct Chain {
    pre: Vec<u32>,
    nxt: Vec<u32>,
}
impl Chain {
    fn new(size: usize) -> Self {
        let end = size as u32;
        Self { pre: vec![end; size], nxt: vec![end; size] }
    }
    fn erase(&mut self, pos: usize)  {
        let len = self.pre.len();
        let end = len as u32;
        let left = self.pre[pos];
        let right = self.nxt[pos];
        if left != end { self.nxt[left as usize] = right; }
        if right != end { self.pre[right as usize] = left; }
        self.pre[pos] = end;
        self.nxt[pos] = end;
    }
}

struct PairIndex {
    minimal_merge_cnt: usize,
    len: usize,
    chain: Chain,
    pair_info: HashMap<Pair, (usize, u32)>, // count, tail
    candidates: HashSet<Pair>,
}
impl PairIndex {
    fn new(size: usize, minimal_merge_cnt: usize) -> Self {
        Self {
            minimal_merge_cnt,
            len: size,
            chain: Chain::new(size),
            pair_info: HashMap::new(),
            candidates: HashSet::new(),
        }
    }

    fn insert(&mut self, pair: Pair, pos: usize, frequency: usize)  {
        let pos_u32 = pos as u32;
        let entry = self.pair_info.entry(pair).or_insert((0, self.len as u32));
        let tail = entry.1;
        self.chain.pre[pos] = tail;
        self.chain.nxt[pos] = self.len as u32;
        if tail != self.len as u32 {
            debug_assert!(tail < pos_u32, "pair occurrences must be appended in text order");
            self.chain.nxt[tail as usize] = pos_u32;
        }
        entry.1 = pos_u32;
        let old_count = entry.0;
        entry.0 += frequency;
        if old_count < self.minimal_merge_cnt && entry.0 >= self.minimal_merge_cnt {
            self.candidates.insert(pair);
        }
    }

    fn erase(&mut self, pair: Pair, pos: usize, frequency: usize)  {
        let pos_u32 = pos as u32;
        let entry = self.pair_info.get_mut(&pair).expect("pair occurrence must exist");
        let previous = self.chain.pre[pos];
        self.chain.erase(pos);
        if entry.1 == pos_u32 { entry.1 = previous; }
        let old_count = entry.0;
        entry.0 -= frequency;
        if old_count >= self.minimal_merge_cnt && entry.0 < self.minimal_merge_cnt {
            self.candidates.remove(&pair);
        }
        if entry.0 == 0 { self.pair_info.remove(&pair); }
    }
}

#[pyfunction]
#[pyo3(signature = (input_path, vocab_size, special_tokens, *, minimal_merge_cnt = 1, **kwargs))]
fn train_bpe(
    py: Python<'_>,
    input_path: PathBuf,
    vocab_size: usize,
    special_tokens: Vec<String>,
    minimal_merge_cnt: usize,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let _ = kwargs;
    if minimal_merge_cnt == 0 {
        py_bail!(PyValueError, "minimal_merge_cnt must be positive");
    }
    let mut special_tokens_unique = Vec::new();
    for token in special_tokens {
        if token.is_empty() { py_bail!(PyValueError, "special tokens must not be empty"); }
        if !special_tokens_unique.contains(&token) { special_tokens_unique.push(token); }
    }
    if vocab_size < 256 + special_tokens_unique.len() || vocab_size > MAX_VOCAB_SIZE {
        py_bail!(PyValueError, "invalid vocabulary size");
    }
    let text = std::fs::read_to_string(input_path)
        .map_err(|error| pyo3::exceptions::PyIOError::new_err(error.to_string()))?;
    let (mut buff, position_frequencies) = pretokenlize(&text, &special_tokens_unique)
        .map_err(|_| pyo3::exceptions::PyOverflowError::new_err("word frequency exceeds u32::MAX"))?;
    drop(text);
    let end = buff.len();
    if end >= u32::MAX as usize {
        py_bail!(PyValueError, "buffer is too large for u32 pair positions");
    }
    let mut pos_chain = Chain::new(end);
    let mut pair_index = PairIndex::new(end, minimal_merge_cnt);
    let mut vocab_map: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
    let mut vocab_merge = Vec::<(Vec<u8>, Vec<u8>)>::new();
    for pos in 1..end {
        if buff[pos - 1] != MAX_VOCAB_SIZE as u16 && buff[pos] != MAX_VOCAB_SIZE as u16 {
            pos_chain.pre[pos] = (pos - 1) as u32;
            pos_chain.nxt[pos - 1] = pos as u32;
            pair_index.insert((buff[pos - 1], buff[pos]), pos, position_frequencies[pos] as usize);
        }
    }
    while vocab_map.len() + special_tokens_unique.len() < vocab_size {
        let selected = pair_index.candidates.iter().copied().max_by(|a, b| {
            pair_index.pair_info[a].0.cmp(&pair_index.pair_info[b].0)
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

        let mut positions = Vec::new();
        let mut pos = pair_index.pair_info[&selected].1 as usize;
        positions.push(pos);
        while pair_index.chain.pre[pos as usize] != pair_index.len as u32 {
            pos = pair_index.chain.pre[pos] as usize;
            positions.push(pos);
        }
        positions.reverse();
        debug_assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        for pos2 in positions {
            let pos1 = pos_chain.pre[pos2] as usize;
            if pos1 == end || (buff[pos1], buff[pos2]) != selected { continue; }
            let frequency = position_frequencies[pos2] as usize;
            let pre_pos = pos_chain.pre[pos1] as usize;
            let nxt_pos = pos_chain.nxt[pos2] as usize;

            if pre_pos != end { pair_index.erase((buff[pre_pos], buff[pos1]), pos1, frequency); }
            pair_index.erase(selected, pos2, frequency);
            if nxt_pos != end { pair_index.erase((buff[pos2], buff[nxt_pos]), nxt_pos, frequency); }

            buff[pos2] = token_id;
            pos_chain.erase(pos1);
            if pre_pos != end { pair_index.insert((buff[pre_pos], token_id), pos2, frequency); }
            if nxt_pos != end { pair_index.insert((token_id, buff[nxt_pos]), nxt_pos, frequency); }
        }
    }
    vocab_map.extend(special_tokens_unique.into_iter().map(String::into_bytes));

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

fn pretokenlize(s: &str, special_tokens: &[String]) -> Result<(Vec<u16>, Vec<u32>), std::num::TryFromIntError> {
    let re = Regex::new(PRETOKENIZE_PAT).expect("pattern should compile");
    let mut words = HashMap::<&[u8], usize>::new();
    let mut pos = 0;
    while pos < s.len() {
        let mut end = s.len();
        let mut next = s.len();
        for pat in special_tokens {
            if let Some(relative) = s[pos..].find(pat.as_str()) {
                let start = pos + relative;
                if start < end || (start == end && start + pat.len() < next) {
                    end = start;
                    next = start + pat.len();
                }
            }
        }
        for matched in re.find_iter(&s[pos..end]) {
            let word = matched.expect("matching should succeed").as_str().as_bytes();
            *words.entry(word).or_default() += 1;
        }
        pos = next;
    }
    let mut buff = vec![MAX_VOCAB_SIZE as u16];
    let mut position_frequencies = vec![0u32];
    for (word, frequency) in words {
        let frequency = u32::try_from(frequency)?;
        buff.extend(word.iter().map(|&byte| byte as u16));
        buff.push(MAX_VOCAB_SIZE as u16);
        position_frequencies.resize(buff.len(), frequency);
    }
    Ok((buff, position_frequencies))
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
    fn pair_index_updates_tail_and_unlinks_occurrences() {
        let pair = (1, 2);
        let mut index = PairIndex::new(10, 1);
        for pos in [1, 3, 5] {
            index.insert(pair, pos, 1);
        }
        index.erase(pair, 3, 1);
        assert_eq!(index.pair_info[&pair], (2, 5));
        assert_eq!(index.chain.nxt[1], 5);
        assert_eq!(index.chain.pre[5], 1);
        assert_eq!(index.pair_info[&pair].1, 5);
        index.erase(pair, 5, 1);
        assert_eq!(index.pair_info[&pair].1, 1);
        assert_eq!(index.pair_info[&pair], (1, 1));
        assert_eq!(index.chain.pre[1], 10);
        index.erase(pair, 1, 1);
        assert!(!index.pair_info.contains_key(&pair));
        assert!(!index.candidates.contains(&pair));
        index.insert((3, 4), 1, 1);
        assert_eq!(index.chain.pre[1], 10);
        assert_eq!(index.chain.nxt[1], 10);
    }
    #[test]
    fn repeated_words_share_buffer_and_preserve_frequency() {
        let specials = vec!["<|endoftext|>".to_string()];
        let (buff, frequencies) = pretokenlize(
            "abab<|endoftext|>abab<|endoftext|>abab", &specials,
        ).unwrap();
        assert_eq!(buff, vec![65535, 97, 98, 97, 98, 65535]);
        assert_eq!(frequencies, vec![0, 3, 3, 3, 3, 3]);
        assert_eq!(frequencies.len(), buff.len());
        let mut index = PairIndex::new(buff.len(), 1);
        index.insert((97, 98), 2, frequencies[2] as usize);
        index.insert((97, 98), 4, frequencies[4] as usize);
        assert_eq!(index.pair_info[&(97, 98)].0, 6);
        index.erase((97, 98), 2, frequencies[2] as usize);
        assert_eq!(index.pair_info[&(97, 98)].0, 3);
    }

    #[test]
    fn weighted_updates_can_jump_across_candidate_threshold() {
        let pair = (1, 2);
        let mut index = PairIndex::new(10, 5);
        index.insert(pair, 1, 3);
        assert!(!index.candidates.contains(&pair));
        index.insert(pair, 3, 4);
        assert!(index.candidates.contains(&pair));
        index.erase(pair, 3, 4);
        assert_eq!(index.pair_info[&pair].0, 3);
        assert!(!index.candidates.contains(&pair));
    }

}
