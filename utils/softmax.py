import numpy as np

class softmax:
    def __init__(self):
        self.x = None
        self.axis = None
        self.out = None

    def forward(self, x, axis=-1):
        """Compute the softmax of x along the given axis."""
        self.x = x
        self.axis = axis
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        self.out = e_x / e_x.sum(axis=axis, keepdims=True)
        return self.out

    def backward(self, grad_output):
        """Compute the gradient of the softmax function.

        Uses the identity dL/dx = s * (grad_output - sum(s * grad_output, axis))
        instead of forming the full Jacobian, so it works for batched inputs.
        """
        s = self.out
        dot = np.sum(grad_output * s, axis=self.axis, keepdims=True)
        return s * (grad_output - dot)
