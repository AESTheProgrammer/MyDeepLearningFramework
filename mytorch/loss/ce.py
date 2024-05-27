from mytorch import Tensor

def CategoricalCrossEntropy(preds: Tensor, label: Tensor) -> Tensor:
    "TODO: implement Categorical Cross Entropy loss"
    return label.__mul__(preds.log()).sum().__neg__()
