import numpy as np
from mytorch import Tensor

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    data = x.data.flatten()
    req_grad = x.requires_grad
    depends_on = x.depends_on
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
