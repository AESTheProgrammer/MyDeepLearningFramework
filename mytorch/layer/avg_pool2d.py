from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class AvgPool2d(Layer):
    def __init__(self, kernel_size=(1, 1), stride=(1, 1), padding=None) -> None:
        super()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        if (self.padding is not None):
            input= Tensor(
                data=initializer([x.shape[0], self.out_channels, x.shape[2] + 2*self.padding[0], x.shape[3] + 2*self.padding[1]], "zero"),
                requires_grad=x.requires_grad
            )
            input[:,:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]] = x
            x = input
        batch_size, channels, height, width = x.data.shape

        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1
        out = initializer([batch_size, channels, out_height, out_width], "zero")

        for l in range(self.out_channels):
            for c in range(x.shape[0]):
                for i in range(out_height):
                    for j in range(out_width):
                        t = x[l, c, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]]
                        t = t.sum()
                        t = t * (1/(self.kernel_size[0]*self.kernel_size[1]))
                        out[l, c, i, j] = t
        return out
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
