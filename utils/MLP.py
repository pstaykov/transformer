import numpy as np
from utils.SwiGLU import SwiGLU
from utils import fp8


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
        self.x = None
        self.dL_dW = None
        self.dL_db = None
        self._Wq = None

    def forward(self, x):
        """x: (..., input_dim) -> (..., output_dim)"""
        self.x = x
        # W doesn't change again until update(), so the quantized copy is
        # cached here and reused in backward() instead of re-quantizing.
        self._Wq = fp8.quantize(self.W) if fp8.ENABLED else self.W
        return fp8.qmatmul(x, self._Wq) + self.b

    def backward(self, dL_dy):
        """dL_dy: (..., output_dim) -> dL_dx: (..., input_dim)

        Batch/sequence dimensions are summed over when accumulating the
        weight and bias gradients.
        """
        reduce_axes = tuple(range(dL_dy.ndim - 1))
        self.dL_dW = np.tensordot(self.x, dL_dy, axes=(reduce_axes, reduce_axes))
        self.dL_db = np.sum(dL_dy, axis=reduce_axes)
        return fp8.qmatmul(dL_dy, self._Wq.T)

    def update(self, lr):
        self.W -= lr * self.dL_dW
        self.b -= lr * self.dL_db

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dL_dW, "b": self.dL_db}

    def load_params(self, d):
        self.W = np.array(d["W"])
        self.b = np.array(d["b"])


class MLP:
    """
    MLP with hardcoded SwiGLU
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden_layer = LinearLayer(n_inputs, 2 * n_hidden)
        self.swiglu = SwiGLU()
        self.output_layer = LinearLayer(n_hidden, n_outputs)

    def forward(self, x):
        hidden_output = self.hidden_layer.forward(x)
        swiglu_hidden_output = self.swiglu.forward(hidden_output)
        output = self.output_layer.forward(swiglu_hidden_output)
        return output

    def backward(self, dL_dyhat):
        grad = self.output_layer.backward(dL_dyhat)
        grad = self.swiglu.backward(grad)
        grad = self.hidden_layer.backward(grad)
        return grad

    def update(self, lr):
        self.hidden_layer.update(lr)
        self.swiglu.update(lr)
        self.output_layer.update(lr)

    def params(self):
        return {
            "hidden_layer": self.hidden_layer.params(),
            "swiglu": self.swiglu.params(),
            "output_layer": self.output_layer.params(),
        }

    def grads(self):
        return {
            "hidden_layer": self.hidden_layer.grads(),
            "swiglu": self.swiglu.grads(),
            "output_layer": self.output_layer.grads(),
        }

    def load_params(self, d):
        self.hidden_layer.load_params(d["hidden_layer"])
        self.swiglu.load_params(d["swiglu"])
        self.output_layer.load_params(d["output_layer"])
