import numpy as np


class SwiGLU:
    """
    SwiGLU(x) = Swish(xW) * (xV)

    Expects a pre-activation input of size 2 * hidden_dim (the concatenation
    of the gate projection and the value projection) and splits it in half.
    """

    def __init__(self, beta=1.0):
        self.beta = beta
        self.gate = None
        self.value = None
        self.sig = None

    def forward(self, x):
        """Compute the SwiGLU activation function."""
        self.gate, self.value = np.split(x, 2, axis=-1)
        self.sig = 1 / (1 + np.exp(-self.beta * self.gate))
        swish_gate = self.gate * self.sig
        return swish_gate * self.value

    def backward(self, grad_output):
        """Compute the gradient of the SwiGLU activation function."""
        swish_gate = self.gate * self.sig
        d_swish = self.sig + self.beta * swish_gate * (1 - self.sig)

        d_gate = grad_output * self.value * d_swish
        d_value = grad_output * swish_gate

        d_swish_dbeta = self.gate * self.gate * self.sig * (1 - self.sig)
        self.dL_dbeta = np.sum(grad_output * self.value * d_swish_dbeta)

        return np.concatenate([d_gate, d_value], axis=-1)

    def update(self, lr):
        self.beta -= lr * self.dL_dbeta
