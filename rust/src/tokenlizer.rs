use std::collections::{BTreeMap, HashMap, HashSet, hash_map};

use fancy_regex::{Match, Regex};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyIterator;

const PRETOKENIZE_PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#[derive(Debug)]
enum TokenlizerError {
    InvalidInput(String),
    Regex(fancy_regex::Error),
}

fn tokenlizer_error_to_py(error: TokenlizerError) -> PyErr {
    match error {
        TokenlizerError::InvalidInput(message) => PyValueError::new_err(message),
        TokenlizerError::Regex(error) =>
            pyo3::exceptions::PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyclass(module = "cs336_basics._rust")]
pub struct Tokenlizer {
    pretokenize_regex: Regex,
    byte_to_token: [u16; 256],
    vocab_map: Vec<Vec<u8>>,
    merge_steps: Vec<(u16, u16, u16)>,
    special_tokens_with_id: Vec<(String, u16)>,
}

#[pyclass(module = "cs336_basics._rust")]
struct EncodeIterator {
    encode: Box<dyn FnMut(Python<'_>, Option<String>) -> Result<Vec<u16>, TokenlizerError> + Send + Sync>,
    input: Option<Py<PyIterator>>,
    pending: std::vec::IntoIter<u16>,
}

#[pymethods]
impl EncodeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<u16>> {
        loop {
            if let Some(token) = self.pending.next() { return Ok(Some(token)); }
            let Some(input) = self.input.as_ref() else { return Ok(None); };
            let item = input.bind(py).clone().next();
            let result = match item {
                None => {
                    self.input = None;
                    (self.encode)(py, None).map_err(tokenlizer_error_to_py)
                }
                Some(item) => item.and_then(|item: Bound<'_, PyAny>| {
                    let text: String = item.extract()?;
                    drop(item);
                    (self.encode)(py, Some(text)).map_err(tokenlizer_error_to_py)
                }),
            };
            match result {
                Ok(tokens) => self.pending = tokens.into_iter(),
                Err(error) => {
                    self.input = None;
                    return Err(error);
                }
            }
        }
    }
}

#[pymethods]
impl Tokenlizer {
    fn encode_iterable(
        slf: Py<Self>,
        iterable: &Bound<'_, PyAny>,
    ) -> PyResult<EncodeIterator> {
        let encode = Self::get_encode_iterable_closure(slf);
        Ok(EncodeIterator {
            encode: Box::new(encode),
            input: Some(iterable.try_iter()?.unbind()),
            pending: Vec::new().into_iter(),
        })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u16>> {
        self.encode_inner(text).map_err(tokenlizer_error_to_py)
    }
    fn decode(&self, ids: Vec<u16>) -> String {
        String::from_utf8_lossy(&self.decode_inner(&ids)).into_owned()
    }

    #[staticmethod]
    #[pyo3(signature = (vocab, merges, special_tokens=None))]
    fn form_vocab(
        vocab: BTreeMap<usize, Vec<u8>>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Self::from_vocab_inner(&vocab, &merges, &special_tokens.unwrap_or_default())
            .map_err(tokenlizer_error_to_py)
    }
}


impl Tokenlizer {
    fn get_encode_iterable_closure(
        tokenizer: Py<Self>,
    ) -> impl FnMut(Python<'_>, Option<String>) -> Result<Vec<u16>, TokenlizerError> + Send + Sync + 'static {
        let mut pre_text = String::new();
        move |py: Python<'_>, append_text: Option<String>| {
            let tokenizer = tokenizer.borrow(py);
            let mut stream_end = false;
            match &append_text {
                None => {stream_end = true}
                Some(append_text) => {pre_text.push_str(&append_text)}
            }
            drop(append_text);
            let text = pre_text.as_str();
            let mut cur_special_token_pos = vec![0usize; tokenizer.special_tokens_with_id.len()];
            let max_special_token_len = tokenizer.special_tokens_with_id
                .iter()
                .map(|(token, _)| token.len())
                .max()
                .unwrap_or(0);
            let mut res = Vec::<u16>::new();
            let mut pos = 0;
            while pos < text.len() {
                let mut end = text.len();
                let mut next = text.len();
                let mut special_token = tokenizer.vocab_map.len() as u16;
                let segment_start = pos;
                for (i, (pat, token_id)) in tokenizer.special_tokens_with_id.iter().enumerate() {
                    let cur_pos = &mut cur_special_token_pos[i];
                    if *cur_pos < pos { *cur_pos = pos; }
                    if let Some(relative) = text[*cur_pos..].find(pat.as_str()) {
                        let start = *cur_pos + relative;
                        if start < end || (start == end && start + pat.len() > next) {
                            end = start;
                            special_token = *token_id;
                            next = start + pat.len();
                        }
                    }
                }
                let mut last_matched: Option<Match> = None;
                let mut solve = |matched: Match| {
                    let mut word : Vec<u16> = 
                        matched.as_str().as_bytes()
                        .iter().map(|v| tokenizer.byte_to_token[*v as usize]).collect();
                    for (x, y, z) in &tokenizer.merge_steps {
                        let mut i = 1;
                        while i < word.len() {
                            if word[i - 1] == *x && word[i] == *y {
                                word.remove(i - 1);
                                word[i - 1] = *z;
                            }
                            else { i += 1; }
                        }
                    }
                    res.extend(word.iter());
                };
                for matched in tokenizer.pretokenize_regex.find_iter(&text[segment_start..end]) {
                    let matched = matched.map_err(TokenlizerError::Regex)?;
                    let last_end = match &last_matched { 
                        None => segment_start, 
                        Some(matched) => matched.end() 
                    };
                    if last_end != segment_start + matched.start() {
                        return Err(TokenlizerError::InvalidInput(
                            "pretokenization left an unmatched gap between matches".into(),
                        ));
                    }
                    if let Some(matched) = last_matched {
                        pos += matched.as_str().len();
                        solve(matched);
                    }
                    last_matched = Some(matched);
                    if !stream_end && end == text.len() 
                    && matched.end() + segment_start + max_special_token_len > end { 
                        break; 
                    }
                }
                if end == text.len() {
                    if let Some(matched) = last_matched {
                        if stream_end { 
                            pos += matched.as_str().len();
                            solve(matched); 
                        }
                        break;
                    }
                    else {
                        assert!(
                            last_matched.is_some(),
                            "pretokenization must produce a match for a nonempty trailing segment"
                        );
                    }
                }
                else {                    
                    if let Some(matched) = last_matched {
                        // pos += matched.as_str().len();
                        solve(matched); 
                    }
                    res.push(special_token);
                    pos = next;
                }
            }
            pre_text = text[pos..].to_owned();
            Ok(res)
        }
    }

    fn from_vocab_inner(
        ordered: &BTreeMap<usize, Vec<u8>>,
        vocab_merge: &Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: &Vec<String>,
    ) -> Result<Self, TokenlizerError> {
        if ordered.len() > usize::from(u16::MAX) + 1 {
            return Err(TokenlizerError::InvalidInput(format!(
                "vocabulary has {} entries; u16 token IDs support at most 65536",
                ordered.len()
            )));
        }
        let mut byte_to_token = [0u16; 256];
        let mut vocab_map = Vec::with_capacity(ordered.len());
        let mut token_ids = HashMap::with_capacity(vocab_map.len());
        for (id, bytes) in ordered {
            if *id != vocab_map.len() {
                return Err(TokenlizerError::InvalidInput(
                    "vocabulary IDs must be contiguous and start at zero".into(),
                ));
            }
            vocab_map.push(bytes.clone());
            let entry = token_ids.entry(bytes);
            if let hash_map::Entry::Vacant(val) = entry {
                val.insert(*id as u16);
            }
            else {
                return Err(TokenlizerError::InvalidInput(format!(
                    "duplicate vocabulary byte sequence at token ID {id}"
                )));
            }
            if bytes.len() == 1 { byte_to_token[bytes[0] as usize] = *id as u16; }
        }
        let mut merge_steps = Vec::with_capacity(vocab_merge.len());
        for (left_bytes, right_bytes) in vocab_merge {
            let left = token_ids.get(&left_bytes).ok_or_else(||
                TokenlizerError::InvalidInput("left merge token is missing from vocabulary".into()))?;
            let right = token_ids.get(&right_bytes).ok_or_else(||
                TokenlizerError::InvalidInput("right merge token is missing from vocabulary".into()))?;
            let mut merged = Vec::with_capacity(left_bytes.len() + right_bytes.len());
            merged.extend_from_slice(&left_bytes);
            merged.extend_from_slice(&right_bytes);
            let result = token_ids.get(&merged).ok_or_else(||
                TokenlizerError::InvalidInput("merge token is missing from vocabulary".into()))?;
            merge_steps.push((*left, *right, *result));
        }
        let mut seen = HashSet::with_capacity(special_tokens.len());
        for token in special_tokens.iter() {
            if !seen.insert(token) {
                return Err(TokenlizerError::InvalidInput(format!(
                    "duplicate special token: {token:?}"
                )));
            }
        }
        drop(seen);
        let mut special_tokens_with_id = Vec::<(String, u16)>::new();
        for pat in special_tokens {
            let id = *token_ids
                .get(&pat.as_bytes().to_vec())
                .expect("special token is missing from vocabulary");
            special_tokens_with_id.push((pat.clone(), id));
        }
        let pretokenize_regex = Regex::new(PRETOKENIZE_PAT).map_err(TokenlizerError::Regex)?;
        Ok(Self { pretokenize_regex, byte_to_token, vocab_map, merge_steps, special_tokens_with_id })
    }

    fn encode_inner(
        &self, 
        text: &str, 
    ) -> Result<Vec<u16>, TokenlizerError> {
        let mut cur_special_token_pos = vec![0usize; self.special_tokens_with_id.len()];
        let mut res = Vec::<u16>::new();
        let mut pos = 0;
        while pos < text.len() {
            let mut end = text.len();
            let mut next = text.len();
            let mut special_token = self.vocab_map.len() as u16;
            for (i, (pat, token_id)) in self.special_tokens_with_id.iter().enumerate() {
                let cur_pos = &mut cur_special_token_pos[i];
                if *cur_pos < pos { *cur_pos = pos; }
                if let Some(relative) = text[*cur_pos..].find(pat.as_str()) {
                    let start = *cur_pos + relative;
                    if start < end || (start == end && start + pat.len() > next) {
                        end = start;
                        special_token = *token_id;
                        next = start + pat.len();
                    }
                }
            }
            let segment_start = pos;
            for matched in self.pretokenize_regex.find_iter(&text[segment_start..end]) {
                let matched = matched.map_err(TokenlizerError::Regex)?;
                if pos != segment_start + matched.start() {
                    return Err(TokenlizerError::InvalidInput(
                        "pretokenization left an unmatched gap between matches".into(),
                    ));
                }
                pos += matched.as_str().len();
                let mut word : Vec<u16> = 
                    matched.as_str().as_bytes()
                    .iter().map(|v| self.byte_to_token[*v as usize]).collect();
                for (x, y, z) in &self.merge_steps {
                    let mut i = 1;
                    while i < word.len() {
                        if word[i - 1] == *x && word[i] == *y {
                            word.remove(i - 1);
                            word[i - 1] = *z;
                        }
                        else { i += 1; }
                    }
                }
                res.extend(word.iter());
            }
            if (special_token as usize) < self.vocab_map.len() { res.push(special_token); }
            pos = next;
        }
        Ok(res)
    }
    fn decode_inner(
        &self,
        input: &Vec<u16>
    ) -> Vec<u8> {
        let mut res = Vec::<u8>::new();
        for token in input {
            res.extend(self.vocab_map[*token as usize].iter());
        }
        res
    }
}

