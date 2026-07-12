import numpy as np


class RMSNorm:
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    y = x / rms(x) * gamma,  rms(x) = sqrt(mean(x**2, axis=-1) + eps)

    Unlike LayerNorm, RMSNorm skips re-centering (no mean subtraction) and
    only rescales by the root-mean-square, which is what LLaMA-style models
    pair with SwiGLU feed-forward blocks.
    """

    def __init__(self, dim=None, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(dim) if dim is not None else None

        self.x = None
        self.xhat = None
        self.rms = None
        self.dgamma = None

    def forward(self, x):
        """
        Args:
            x: array of shape (..., d_model)

        Returns:
            Normalized + rescaled array, same shape as x.
        """
        if self.gamma is None:
            self.gamma = np.ones(x.shape[-1])

        self.x = x
        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        self.rms = np.sqrt(ms + self.eps)
        self.xhat = x / self.rms
        return self.xhat * self.gamma

    def backward(self, grad_output):
        """Compute dL/dx and accumulate dL/dgamma."""
        n = self.x.shape[-1]

        reduce_axes = tuple(range(grad_output.ndim - 1))
        self.dgamma = np.sum(grad_output * self.xhat, axis=reduce_axes)

        dxhat = grad_output * self.gamma
        mean_term = np.mean(dxhat * self.xhat, axis=-1, keepdims=True)
        dx = (dxhat - self.xhat * mean_term) / self.rms
        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma

    def params(self):
        return {"gamma": self.gamma}

    def grads(self):
        return {"gamma": self.dgamma}

    def load_params(self, d):
        self.gamma = np.array(d["gamma"])
