import pytest

from cs336_basics._rust import Tokenlizer


@pytest.fixture
def tokenizer():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"<END>"
    return Tokenlizer.form_vocab(vocab, [(b"a", b"b")], ["<END>"])


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        ["", ""],
        ["a"],
        ["a", "b"],
        ["ab cd"],
        ["ab", " ", "cd"],
        ["<EN", "D>"],
        ["ab<END>"],
        ["ab<END>", "cd"],
        ["你", "好", "🙂"],
        ["a", "", "b"],
    ],
)
def test_stream_matches_whole_text(tokenizer, chunks):
    assert list(tokenizer.encode_iterable(iter(chunks))) == tokenizer.encode(
        "".join(chunks)
    )


def test_stream_is_lazy_and_stays_exhausted(tokenizer):
    consumed = []

    def source():
        consumed.append(True)
        yield "a"

    output = tokenizer.encode_iterable(source())
    assert consumed == []
    assert list(output) == tokenizer.encode("a")
    assert consumed == [True]
    assert next(output, None) is None
    assert next(output, None) is None
