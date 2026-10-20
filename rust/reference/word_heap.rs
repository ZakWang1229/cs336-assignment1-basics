use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

const MAX_VOCAB_SIZE: usize = 65535;
const PRETOKENIZE_PAT: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

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
    let buff = pretokenlize(&text, specials.clone());
    let mut frequencies = HashMap::<Vec<u16>, usize>::new();
    for word in buff.split(|&token| token == MAX_VOCAB_SIZE as u16) {
        if !word.is_empty() {
            *frequencies.entry(word.to_vec()).or_default() += 1;
        }
    }
    let mut words: Vec<(Vec<u16>, usize)> = frequencies.into_iter().collect();
    let mut vocab_map: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
    let mut vocab_merge = Vec::<(Vec<u8>, Vec<u8>)>::new();
    type Pair = (u16, u16);
    let mut counts = HashMap::<Pair, usize>::new();
    let mut locations = HashMap::<Pair, HashSet<usize>>::new();
    for (id, (word, frequency)) in words.iter().enumerate() {
        for window in word.windows(2) {
            let pair = (window[0], window[1]);
            *counts.entry(pair).or_default() += frequency;
            locations.entry(pair).or_default().insert(id);
        }
    }
    // Heap entries carry byte strings so ties follow the required byte ordering.
    let mut heap = std::collections::BinaryHeap::new();
    for (&pair, &count) in &counts {
        heap.push((
            count,
            vocab_map[pair.0 as usize].clone(),
            vocab_map[pair.1 as usize].clone(),
            pair,
        ));
    }
    while vocab_map.len() + specials.len() < vocab_size {
        let Some((count, _, _, selected)) = heap.pop() else {
            break;
        };
        if count == 0 || counts.get(&selected).copied().unwrap_or(0) != count {
            continue;
        }
        let left = vocab_map[selected.0 as usize].clone();
        let right = vocab_map[selected.1 as usize].clone();
        let token_id = vocab_map.len() as u16;
        let mut merged = left.clone();
        merged.extend_from_slice(&right);
        vocab_map.push(merged);
        vocab_merge.push((left, right));
        let affected = locations.remove(&selected).unwrap_or_default();
        let mut changed = HashSet::new();
        for id in affected {
            let (word, frequency) = &mut words[id];
            for window in word.windows(2) {
                let pair = (window[0], window[1]);
                *counts.get_mut(&pair).expect("existing pair count") -= *frequency;
                if let Some(ids) = locations.get_mut(&pair) {
                    ids.remove(&id);
                }
                changed.insert(pair);
            }
            let mut output = Vec::with_capacity(word.len());
            let mut position = 0;
            while position < word.len() {
                if position + 1 < word.len() && (word[position], word[position + 1]) == selected {
                    output.push(token_id);
                    position += 2;
                } else {
                    output.push(word[position]);
                    position += 1;
                }
            }
            *word = output;
            for window in word.windows(2) {
                let pair = (window[0], window[1]);
                *counts.entry(pair).or_default() += *frequency;
                locations.entry(pair).or_default().insert(id);
                changed.insert(pair);
            }
        }
        for pair in changed {
            let count = counts[&pair];
            if count > 0 {
                heap.push((
                    count,
                    vocab_map[pair.0 as usize].clone(),
                    vocab_map[pair.1 as usize].clone(),
                    pair,
                ));
            }
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
}
