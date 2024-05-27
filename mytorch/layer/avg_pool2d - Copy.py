from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        ks_h, ks_w = self.kernel_size[0], self.kernel_size[1]
        pd_h, pd_w = self.padding[0], self.padding[1]
        st_h, st_w = self.stride[0], self.stride[1]
        out_h = (x.shape[1] + 2 * pd_w  - ks_w)//st_w + 1 
        out_w = (x.shape[2] + 2 * pd_h  - ks_h)//st_h + 1 
        output = Tensor(np.zeros((self.out_channels, out_h, out_w)))
        padded_x = Tensor(np.zeros((self.in_channels, x.shape[1]+pd_h*2, x.shape[2]+pd_w*2)))
        padded_x[:, pd_h:padded_x.shape[1]-pd_h, pd_w:padded_x.shape[2]-pd_w] = x
        kernel = Tensor(np.ones((ks_h, ks_w))/(ks_w*ks_h))
        for oc in range(self.out_channels):  # Loop through each output channel
            for y in range(out_h):  # Loop through output height
                for i in range(out_w):  # Loop through output width
                    # Extract input sub-window
                    input_window = padded_x[oc, y * st_h:y * st_h + ks_h, i * st_w:i * st_w + ks_w]
                    # Element-wise multiplication and summation
                    output[oc, y, i] = (input_window * kernel).sum()
        return output
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
