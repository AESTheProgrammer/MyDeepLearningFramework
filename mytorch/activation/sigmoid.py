import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    data = 1/(1 + np.exp(x.data))
    requires_grad = x.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * ((1 - data) * data)
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)