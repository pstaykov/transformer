import numpy as np
from utils.softmax import softmax


class AttnHead:
    """Single-head scaled dot-product attention.

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
            Q: Query matrix of shape (B, D, D_head)
            K: Key matrix of shape (B, D, D_head)
            V: Value matrix of shape (B, D, D_head)
            M: Bias matrix of shape (B, D, D)
            mask: Optional mask of shape (B, D, D)
        """
        self.Q, self.K, self.V, self.M, self.mask = Q, K, V, M, mask
        self.d_head = Q.shape[-1]

        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_head) + M

        if mask is not None:
            scores = scores + mask * -1e9

        self.attention_weights = self.softmax.forward(scores, axis=-1)
        output = np.matmul(self.attention_weights, V)

        return output, self.attention_weights

    def backward(self, dL_doutput):
        """Compute the gradients of Q, K, V, M given the gradient of the output."""
        dL_dV = np.matmul(self.attention_weights.transpose(0, 2, 1), dL_doutput)
        dL_dattn = np.matmul(dL_doutput, self.V.transpose(0, 2, 1))

        dL_dscores = self.softmax.backward(dL_dattn)
        dL_dM = dL_dscores

        scale = 1.0 / np.sqrt(self.d_head)
        dL_dQ = np.matmul(dL_dscores, self.K) * scale
        dL_dK = np.matmul(dL_dscores.transpose(0, 2, 1), self.Q) * scale

        return dL_dQ, dL_dK, dL_dV, dL_dM
