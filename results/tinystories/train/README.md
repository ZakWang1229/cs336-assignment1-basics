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

- [Vocabulary](vocab.pkl): `dict[int, bytes]`.
- [Ordered merges](merges.pkl): `list[tuple[bytes, bytes]]`.
- [Run statistics](summary.json).

The full training dataset remains under `data/` and is excluded from Git.
