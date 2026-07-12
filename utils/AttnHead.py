import numpy as np
from utils.softmax import softmax
from utils import fp8


class AttnHead:
    """Scaled dot-product attention, vectorized across an arbitrary batch of
    leading dimensions (e.g. (B, T, d_head) for one head, or (B, H, T, d_head)
    for all heads at once via a single batched matmul).

    mask looks like this for example:
    [[1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1]]
    """

    def __init__(self):
        self.softmax = softmax()
        self.Q = None
        self.K = None
        self.V = None
        self.M = None
        self.mask = None
        self.d_head = None
        self.attention_weights = None

    def forward(self, Q, K, V, M, mask=None):
        """Compute the attention output.

        Args:
            Q: Query array of shape (..., T, d_head)
            K: Key array of shape (..., T, d_head)
            V: Value array of shape (..., T, d_head)
            M: Bias array broadcastable to (..., T, T)
            mask: Optional mask broadcastable to (..., T, T)
        """
        self.Q, self.K, self.V, self.M, self.mask = Q, K, V, M, mask
        self.d_head = Q.shape[-1]

        scores = fp8.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_head) + M

        if mask is not None:
            scores = scores + mask * -1e9

        self.attention_weights = self.softmax.forward(scores, axis=-1)
        output = fp8.matmul(self.attention_weights, V)

        return output, self.attention_weights

    def backward(self, dL_doutput):
        """Compute the gradients of Q, K, V, M given the gradient of the output."""
        dL_dV = fp8.matmul(self.attention_weights.swapaxes(-1, -2), dL_doutput)
        dL_dattn = fp8.matmul(dL_doutput, self.V.swapaxes(-1, -2))

        dL_dscores = self.softmax.backward(dL_dattn)
        dL_dM = dL_dscores

        scale = 1.0 / np.sqrt(self.d_head)
        dL_dQ = fp8.matmul(dL_dscores, self.K) * scale
        dL_dK = fp8.matmul(dL_dscores.swapaxes(-1, -2), self.Q) * scale

        return dL_dQ, dL_dK, dL_dV, dL_dM
