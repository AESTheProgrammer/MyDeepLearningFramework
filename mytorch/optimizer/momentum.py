from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement Momentum optimizer"
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum

        for layer in self.layers:
            layer.weight_velocity = np.zeros_like(layer.weight.data)
            if layer.need_bias:
                layer.bias_velocity = np.zeros_like(layer.bias.data)

    def step(self):
        for layer in self.layers:
            layer.weight_velocity = self.momentum * layer.weight_velocity - self.learning_rate * layer.weight.grad
            layer.weight.data = layer.weight.data + layer.weight_velocity

            if layer.need_bias:
                layer.bias_velocity = self.momentum * layer.bias_velocity - self.learning_rate * layer.bias.grad
                layer.bias.data = layer.bias.data + layer.bias_velocity