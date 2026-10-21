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
