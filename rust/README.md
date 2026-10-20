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

`rust/src/lib.rs` provides the interface only and raises `NotImplementedError`
until you implement the training algorithm. The expected Python return value
is a tuple containing `dict[int, bytes]` and `list[tuple[bytes, bytes]]`.
