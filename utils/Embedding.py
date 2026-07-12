import numpy as np


class Embedding:
    """Token + learned positional embeddings."""

    def __init__(self, vocab_size, d_model, max_len):
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_len, d_model) * 0.02

        self.token_ids = None
        self.T = None
        self.d_token_emb = None
        self.d_pos_emb = None

    def forward(self, token_ids):
        """
        Args:
            token_ids: int array of shape (B, T)

        Returns:
            Array of shape (B, T, d_model)
        """
        self.token_ids = token_ids
        self.T = token_ids.shape[1]
        return self.token_emb[token_ids] + self.pos_emb[:self.T]

    def backward(self, dL_dout):
        """dL_dout: (B, T, d_model). Embedding is a leaf, so no dL/dx is returned."""
        self.d_token_emb = np.zeros_like(self.token_emb)
        np.add.at(self.d_token_emb, self.token_ids, dL_dout)

        self.d_pos_emb = np.zeros_like(self.pos_emb)
        self.d_pos_emb[:self.T] = np.sum(dL_dout, axis=0)

    def update(self, lr):
        self.token_emb -= lr * self.d_token_emb
        self.pos_emb -= lr * self.d_pos_emb

    def params(self):
        return {"token_emb": self.token_emb, "pos_emb": self.pos_emb}

    def grads(self):
        return {"token_emb": self.d_token_emb, "pos_emb": self.d_pos_emb}

    def load_params(self, d):
        self.token_emb = np.array(d["token_emb"])
        self.pos_emb = np.array(d["pos_emb"])
