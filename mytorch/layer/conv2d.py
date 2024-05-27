from mytorch import Tensor
from mytorch.layer import Layer
from math import floor
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        ks_h, ks_w = self.kernel_size[0], self.kernel_size[1]
        pd_h, pd_w = self.padding[0], self.padding[1]
        st_h, st_w = self.stride[0], self.stride[1]
        assert x.shape[1] == self.in_channels, "x.shape[1] and in_channels mismatch."
        assert 4 == len(x.shape) # used for handling 2d inputs (instead of making them 3d we use this number)
        out_w = (x.shape[2] + 2 * pd_w  - ks_w)//st_w + 1 
        out_h = (x.shape[3] + 2 * pd_h  - ks_h)//st_h + 1 
        output = Tensor(np.zeros((x.shape[0], self.out_channels, out_h, out_w)), requires_grad=True)
        # batch size/ input channel / H / W
        if self.padding[0] + self.padding[1] > 0:
            padded_x = Tensor(np.zeros((x.shape[0], x.shape[1], x.shape[2]+pd_h*2, x.shape[3]+pd_w*2)), requires_grad=x.requires_grad)
            padded_x[:, :, pd_h:padded_x.shape[2]-pd_h, pd_w:padded_x.shape[3]-pd_w] = x
        else:
            padded_x = x
        for oc in range(self.out_channels):  # Loop through each output channel
            # print("oc: ", oc)
            for y in range(out_h):  # Loop through output height
                for i in range(out_w):  # Loop through output width
                    # Extract input sub-window
                    input_window = padded_x[:, :, y * st_h:y * st_h + ks_h, i * st_w:i * st_w + ks_w] * self.weight[oc]
                    for batch_index in range(x.shape[0]):
                        output[batch_index, oc, y, i] = input_window[batch_index].sum()
        if self.need_bias:
            output = output + self.bias
        return output
    
    
    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer([self.out_channels, self.in_channels, *self.kernel_size], self.initialize_mode),
            requires_grad=True
        )
        if self.need_bias:
            self.bias = Tensor(
                data=initializer([self.out_channels], "zero"),
                requires_grad=True
            )
        else:
            self.bias = Tensor(
                data=initializer([self.out_channels], "zero"),
                requires_grad=False
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()
        pass

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return (self.weight, self.bias)
        return self.weight
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)