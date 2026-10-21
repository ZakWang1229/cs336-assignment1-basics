# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

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

The final artifacts are checked into this repository (about 243 KiB total):

- [Vocabulary](results/tinystories/vocab.pkl): `dict[int, bytes]`.
- [Ordered merges](results/tinystories/merges.pkl): `list[tuple[bytes, bytes]]`.
- [Run statistics](results/tinystories/summary.json).

The full training dataset remains under `data/` and is excluded from Git.
