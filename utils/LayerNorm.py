import numpy as np

def mean(x, axis=None, keepdims=False):
    """Compute the mean of array x along the specified axis."""
    return np.mean(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    """Compute the standard deviation of array x along the specified axis."""
    return np.std(x, axis=axis, keepdims=keepdims)

class LayerNorm:
    def __init__(self):
        self.epsilon = 1e-5
        self.x = None

    def forward(self, x):
        """Apply Layer Normalization to the input array x.

        Args:
            x: Input array of shape (B, D, D_head)

        Returns:
            Normalized array of the same shape as x
        """
        self.x = x
        self.n = x.shape[-1]
        self.mean_x = mean(x, axis=-1, keepdims=True)
        self.std_x = std(x, axis=-1, keepdims=True)
        self.denom = self.std_x + self.epsilon
        self.out = (x - self.mean_x) / self.denom
        return self.out

    def backward(self, grad_output):
        """Compute the gradient of Layer Normalization w.r.t. its input."""
        grad_mean = mean(grad_output, axis=-1, keepdims=True)
        grad_dot_out = np.sum(grad_output * self.out, axis=-1, keepdims=True)

        return (grad_output - grad_mean) / self.denom - (
            self.out * grad_dot_out
        ) / (self.n * self.std_x)