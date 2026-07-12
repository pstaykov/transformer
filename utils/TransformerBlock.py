from utils.MultiHeadAttention import MultiHeadAttention
from utils.RMSNorm import RMSNorm
from utils.MLP import MLP


class TransformerBlock:
    """Pre-norm transformer block: attention + SwiGLU MLP, each with a residual connection.

    Uses RMSNorm (LLaMA-style) rather than LayerNorm ahead of each sub-layer.
    """

    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, dL_dout):
        """dL_dout: (B, T, d_model) -> dL_dx: (B, T, d_model)

        Each sub-layer is wrapped in a residual (x = x + sublayer(norm(x))), so
        the gradient splits into an identity path and a sublayer path at every stage.
        """
        dL_dln2_out = self.mlp.backward(dL_dout)
        dL_dx1_from_mlp = self.ln2.backward(dL_dln2_out)
        dL_dx1 = dL_dout + dL_dx1_from_mlp

        dL_dln1_out = self.attn.backward(dL_dx1)
        dL_dx0_from_attn = self.ln1.backward(dL_dln1_out)
        dL_dx0 = dL_dx1 + dL_dx0_from_attn

        return dL_dx0

    def update(self, lr):
        self.attn.update(lr)
        self.ln1.update(lr)
        self.mlp.update(lr)
        self.ln2.update(lr)

    def params(self):
        return {
            "attn": self.attn.params(),
            "ln1": self.ln1.params(),
            "mlp": self.mlp.params(),
            "ln2": self.ln2.params(),
        }

    def grads(self):
        return {
            "attn": self.attn.grads(),
            "ln1": self.ln1.grads(),
            "mlp": self.mlp.grads(),
            "ln2": self.ln2.grads(),
        }

    def load_params(self, d):
        self.attn.load_params(d["attn"])
        self.ln1.load_params(d["ln1"])
        self.mlp.load_params(d["mlp"])
        self.ln2.load_params(d["ln2"])
