from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer
from mytorch import Tensor
import numpy as np

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implementreuturn  SGD algorithm"

        for layer in self.layers:
            layer.weight = layer.weight - self.learning_rate * layer.weight.grad
            if layer.need_bias:
                layer.bias = layer.bias - self.learning_rate * layer.bias.grad



