import numpy as np
from utils.AttnHead import AttnHead
from utils import fp8


class MultiHeadAttention:
    """Multi-head attention. All heads are computed in one batched AttnHead
    call over a (B, H, T, d_head) tensor rather than looping per head - numpy's
    matmul already broadcasts over leading batch dims, and looping in Python
    just multiplies the number of (relatively expensive) FP8 quantize calls
    by the head count for no benefit.
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, d_model) * 0.1
        self.Wv = np.random.randn(d_model, d_model) * 0.1
        self.Wo = np.random.randn(d_model, d_model) * 0.1

        self.attn = AttnHead()

        self.x = None
        self.merged = None
        self.dWq = self.dWk = self.dWv = self.dWo = None
        self._Wq_q = self._Wk_q = self._Wv_q = self._Wo_q = None

    def _split_heads(self, x):
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (B, H, T, d_head)

    def _merge_heads(self, x):
        B, H, T, d_head = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(B, T, H * d_head)

    def forward(self, x, mask=None):
        """
        Args:
            x: Array of shape (B, T, d_model)
            mask: Optional mask of shape (B, T, T) or broadcastable to it

        Returns:
            Array of shape (B, T, d_model)
        """
        B, T, _ = x.shape
        self.x = x

        # x and the projection weights are each quantized once and reused
        # (x across all 3 projections here, weights again in backward())
        # instead of re-quantizing the same array on every matmul.
        xq = fp8.quantize(x) if fp8.ENABLED else x
        self._Wq_q = fp8.quantize(self.Wq) if fp8.ENABLED else self.Wq
        self._Wk_q = fp8.quantize(self.Wk) if fp8.ENABLED else self.Wk
        self._Wv_q = fp8.quantize(self.Wv) if fp8.ENABLED else self.Wv

        Qh = self._split_heads(fp8.qmatmul(xq, self._Wq_q))
        Kh = self._split_heads(fp8.qmatmul(xq, self._Wk_q))
        Vh = self._split_heads(fp8.qmatmul(xq, self._Wv_q))

        M = np.zeros((B, 1, T, T))
        mask4 = mask[:, None, :, :] if mask is not None else None

        out, _ = self.attn.forward(Qh, Kh, Vh, M, mask4)  # (B, H, T, d_head)

        self.merged = self._merge_heads(out)
        self._Wo_q = fp8.quantize(self.Wo) if fp8.ENABLED else self.Wo
        return fp8.qmatmul(self.merged, self._Wo_q)

    def backward(self, dL_dout):
        """dL_dout: (B, T, d_model) -> dL_dx: (B, T, d_model)"""
        self.dWo = np.tensordot(self.merged, dL_dout, axes=([0, 1], [0, 1]))
        dL_dmerged = fp8.qmatmul(dL_dout, self._Wo_q.T)

        B, T, _ = dL_dmerged.shape
        dL_dmerged_heads = dL_dmerged.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        dQh, dKh, dVh, _ = self.attn.backward(dL_dmerged_heads)  # each (B, H, T, d_head)

        dQ_merged = self._merge_heads(dQh)
        dK_merged = self._merge_heads(dKh)
        dV_merged = self._merge_heads(dVh)

        self.dWq = np.tensordot(self.x, dQ_merged, axes=([0, 1], [0, 1]))
        self.dWk = np.tensordot(self.x, dK_merged, axes=([0, 1], [0, 1]))
        self.dWv = np.tensordot(self.x, dV_merged, axes=([0, 1], [0, 1]))

        dx = (
            fp8.qmatmul(dQ_merged, self._Wq_q.T)
            + fp8.qmatmul(dK_merged, self._Wk_q.T)
            + fp8.qmatmul(dV_merged, self._Wv_q.T)
        )
        return dx

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo

    def params(self):
        return {"Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv, "Wo": self.Wo}

    def grads(self):
        return {"Wq": self.dWq, "Wk": self.dWk, "Wv": self.dWv, "Wo": self.dWo}

    def load_params(self, d):
        self.Wq = np.array(d["Wq"])
        self.Wk = np.array(d["Wk"])
        self.Wv = np.array(d["Wv"])
        self.Wo = np.array(d["Wo"])
