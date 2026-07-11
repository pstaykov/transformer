"""Demo of the bbpe_tokenizer Python bindings."""

import bbpe_tokenizer

TOKENIZER_PATH = "tokenizer/tok_out/tokenizer.bbpe"


def main():
    tok = bbpe_tokenizer.Tokenizer.load_binary(TOKENIZER_PATH)

    samples = [
        "Hello world, this is a test!",
        "Byte-pair encoding compresses common substrings.",
        "1234 + 5678 = 6912",
    ]

    for text in samples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)

        print(f"Text        : {text}")
        print(f"Token IDs   : {ids}")
        print(f"Num tokens  : {len(ids)}")
        print(f"Decoded     : {decoded}")
        print(f"Roundtrip OK: {decoded == text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
