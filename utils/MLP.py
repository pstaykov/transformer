import numpy as np
from utils.SwiGLU import SwiGLU

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = float(np.zeros_like(output_dim))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dL_dy):
        self.dL_dW = np.outer(self.x, dL_dy)
        self.dL_db = dL_dy
        return np.dot(dL_dy, self.W.T)

    def update(self, lr):
        self.W -= lr * self.dL_dW
        self.b -= lr * self.dL_db

class MLP():
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