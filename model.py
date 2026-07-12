import numpy as np
from utils.Embedding import Embedding
from utils.TransformerBlock import TransformerBlock
from utils.RMSNorm import RMSNorm
from utils import fp8


class Transformer:
    """Decoder-only transformer (forward + backward), RMSNorm + SwiGLU feed-forward blocks."""

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=24, d_ff=2026, max_len=512):
        # d_model/d_ff picked so each TransformerBlock has ~100_000_000 / 24
        # parameters (4*d_model^2 attn + ~3*d_model*d_ff SwiGLU MLP).
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_f = RMSNorm(d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.02

        self._ln_f_out = None
        self.dW_out = None
        self._W_out_q = None

    def _causal_mask(self, T):
        """1 marks positions that must be masked out (future tokens)."""
        return np.triu(np.ones((T, T)), k=1)[None, :, :]

    def forward(self, token_ids):
        """
        Args:
            token_ids: int array of shape (B, T)

        Returns:
            logits of shape (B, T, vocab_size)
        """
        _, T = token_ids.shape
        mask = self._causal_mask(T)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)
        self._ln_f_out = x

        self._W_out_q = fp8.quantize(self.W_out) if fp8.ENABLED else self.W_out
        return fp8.qmatmul(x, self._W_out_q)

    def backward(self, dL_dlogits):
        """dL_dlogits: (B, T, vocab_size), gradient of the loss w.r.t. the output logits."""
        self.dW_out = np.tensordot(self._ln_f_out, dL_dlogits, axes=([0, 1], [0, 1]))
        dx = fp8.qmatmul(dL_dlogits, self._W_out_q.T)

        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embedding.backward(dx)

    def update(self, lr):
        self.embedding.update(lr)
        for block in self.blocks:
            block.update(lr)
        self.ln_f.update(lr)
        self.W_out -= lr * self.dW_out

    def params(self):
        return {
            "embedding": self.embedding.params(),
            "blocks": {str(i): b.params() for i, b in enumerate(self.blocks)},
            "ln_f": self.ln_f.params(),
            "W_out": self.W_out,
        }

    def grads(self):
        return {
            "embedding": self.embedding.grads(),
            "blocks": {str(i): b.grads() for i, b in enumerate(self.blocks)},
            "ln_f": self.ln_f.grads(),
            "W_out": self.dW_out,
        }

    def load_params(self, d):
        self.embedding.load_params(d["embedding"])
        for i, b in enumerate(self.blocks):
            b.load_params(d["blocks"][str(i)])
        self.ln_f.load_params(d["ln_f"])
        self.W_out = np.array(d["W_out"])


if __name__ == "__main__":
    import time
    import bbpe_tokenizer

    print("Warming up numba JIT for FP8 quantization...")
    t_warmup = time.time()
    fp8.warmup()
    print(f"  done in {time.time() - t_warmup:.2f}s\n")

    tok = bbpe_tokenizer.Tokenizer.load_binary("tokenizer/tok_out/tokenizer.bbpe")
    vocab_size = 32000

    texts = [
        "Hello world, this is a test!",
        "Byte-pair encoding compresses common substrings.",
    ]
    encoded = [tok.encode(t) for t in texts]
    max_len = max(len(ids) for ids in encoded)
    token_ids = np.array([ids + [0] * (max_len - len(ids)) for ids in encoded])

    model = Transformer(vocab_size, max_len=max_len)

    from utils.checkpoint import count_params
    block_params = count_params(model.blocks[0].params())
    total_params = count_params(model.params())
    print(f"params per block : {block_params:,}")
    print(f"total blocks     : {len(model.blocks)}")
    print(f"total block params: {block_params * len(model.blocks):,}")
    print(f"total model params (incl. embeddings/output): {total_params:,}")
    print()

    # --- full forward pass ---
    logits = model.forward(token_ids)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    predicted_ids = np.argmax(logits, axis=-1)

    print(f"input shape      : {token_ids.shape}")
    print(f"logits shape     : {logits.shape}")
    print(f"probs shape      : {probs.shape}")
    print(f"predicted shape  : {predicted_ids.shape}")
    print()

    for i, text in enumerate(texts):
        pred_text = tok.decode(predicted_ids[i].tolist())
        top_conf = probs[i, np.arange(max_len), predicted_ids[i]]
        print(f"input text        : {text}")
        print(f"input token ids   : {token_ids[i].tolist()}")
        print(f"predicted next ids: {predicted_ids[i].tolist()}")
        print(f"predicted (decoded, untrained weights): {pred_text!r}")
        print(f"max softmax prob per position: {np.round(top_conf, 4).tolist()}")
        print("-" * 60)
