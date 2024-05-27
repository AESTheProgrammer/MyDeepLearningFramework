from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement RMSprop optimizer"
class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.1, alpha=0.99, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        for layer in self.layers:
            layer.weight_sq_avg = np.zeros_like(layer.weight.data)
            if layer.need_bias:
                layer.bias_sq_avg = np.zeros_like(layer.bias.data)

    def step(self):
        for layer in self.layers:
            layer.weight_sq_avg = self.alpha * layer.weight_sq_avg + \
                (1 - self.alpha) * layer.weight.grad**2
            layer.weight.data = layer.weight.data - self.learning_rate * \
                (layer.weight.grad / (layer.weight_sq_avg**2 + self.epsilon))
            if layer.need_bias:
                layer.bias_sq_avg = self.alpha * layer.bias_sq_avg + \
                    (1 - self.alpha) *layer.bias.grad**2
                layer.bias.data = layer.bias.data - self.learning_rate * \
                    (layer.bias.grad / (layer.bias_sq_avg**0.5 + self.epsilon))