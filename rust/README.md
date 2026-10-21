# Rust extension

The project uses maturin and PyO3 to build `cs336_basics._rust`.
Run these commands from the repository root:

```sh
uv sync
uv run python -c 'from cs336_basics import _rust; print(_rust.__name__)'
```

After editing Rust sources, rebuild the extension:

```sh
uv sync --reinstall-package cs336_basics
```

For optimized builds:

```sh
uv run --with maturin maturin develop --release --manifest-path rust/Cargo.toml
```

`tests/adapters.py::run_train_bpe` forwards calls to the exported Rust
`train_bpe` function, including extra keyword arguments. Paths are normalized
with `os.fspath` before crossing the boundary.

`rust/src/lib.rs` implements byte-level BPE training and returns
`tuple[dict[int, bytes], list[tuple[bytes, bytes]]]`. Repeated pre-tokens are
stored once with weighted frequencies. Token and pair occurrence lists use
`u32` links. The optional keyword argument `minimal_merge_cnt` defaults to 1.
The current test adapter explicitly passes 100; this can stop training early
on small corpora and need not match snapshots generated with threshold 1.

## Full TinyStories training result

Run date: 2026-10-21. Release build on the complete
`data/TinyStoriesV2-GPT4-train.txt` (2,227,753,162 bytes), rather than the 5 MB
pytest fixture.

| Setting or measurement | Result |
| --- | --- |
| Requested vocabulary size | 10,000 |
| Special tokens | `<\|endoftext\|>` |
| Minimum weighted merge count | 100 |
| Training wall time | 138.32 seconds |
| Peak process RSS | 2.1033 GiB |
| Actual vocabulary size | 10,000 |
| Merge count | 9,743 |
| Longest token | `b' accomplishment'` (15 bytes) |

The run completed successfully below the 30 GiB memory budget. Runtime checks
verified that the special token was present, was excluded from merges, and
that vocabulary and merge counts were consistent. This full-corpus run is
not a comparison against a reference vocabulary or a full assignment test run.

Local artifacts are in `data/tinystories_bpe_full/`: `vocab.pkl`, `merges.pkl`,
and `summary.json`. The dataset and generated artifacts are excluded from Git.
