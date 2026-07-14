"""Tokenizer selection shared by the trainer and the inference server.

The real tokenizer is the from-scratch C++ byte-level BPE in tokenizer/, exposed
to Python as the `bbpe_tokenizer` extension module (built by
tokenizer/build_py_client.sh). When it isn't importable we fall back to a plain
byte-level tokenizer so everything still runs with nothing but numpy.
"""

try:
    import bbpe_tokenizer
    HAS_BBPE = True
except ImportError:
    HAS_BBPE = False


class ByteTokenizer:
    """Fallback tokenizer: every byte of UTF-8 text is its own token."""

    vocab_size = 257  # 256 byte values + 1 pad id
    pad_id = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(b for b in ids if b < 256).decode("utf-8", errors="replace")


def load_tokenizer(kind, path=None, vocab_size=32000):
    """Returns (tokenizer, vocab_size) for kind in {"bbpe", "byte"}."""
    if kind == "bbpe":
        if not HAS_BBPE:
            raise RuntimeError(
                "bbpe_tokenizer is not importable in this environment; "
                "run with the tokenizer/venv interpreter, or use the byte tokenizer."
            )
        return bbpe_tokenizer.Tokenizer.load_binary(path), vocab_size
    tok = ByteTokenizer()
    return tok, tok.vocab_size
