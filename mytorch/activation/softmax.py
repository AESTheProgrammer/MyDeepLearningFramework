import numpy as np
from mytorch import Tensor, Dependency
from mytorch.util import flatten


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    max_values = Tensor(np.max(x.data, axis=1, keepdims=True))
    x = x - max_values
    exp = x.exp()
    exp_sum = exp.__matmul__(np.ones((exp.shape[-1], 1)))
    softmax = exp.__mul__(exp_sum.__pow__(-1))
    return softmax
