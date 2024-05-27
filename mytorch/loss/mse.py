from mytorch import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor) -> Tensor:
    "TODO: implement Mean Squared Error loss"
    subs_sum = preds.__sub__(actual)
    subs_sum = subs_sum.__pow__(2)
    subs_sum = subs_sum.sum()
    subs_sum = subs_sum * Tensor(np.array(1/preds.shape[0]))
    return subs_sum
