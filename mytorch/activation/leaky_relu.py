import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor, leak=0.01) -> Tensor:
    """
    TODO: implement leaky_relu function.
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """
    # print(x.data)
    data = np.maximum(leak * x.data, x.data)
    # print(data[0][0])
    # print(data)
    req_grad = x.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(data > 0, 1, leak)
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
