from .tensor import Tensor
import random
import cpp_backend


class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(
            [[random.uniform(-0.1, 0.1) for _ in range(out_features)]
             for _ in range(in_features)],
            requires_grad=True
        )
        self.b = Tensor(
            [[0.0 for _ in range(out_features)]],   # 2D bias
            requires_grad=True
        )

    def __call__(self, x):
        out_data = cpp_backend.matmul_forward(x.data, self.W.data, self.b.data)
        out = Tensor(out_data, requires_grad=True)

        def _backward():
            B = len(x.data)
            in_feat = len(self.W.data)
            out_feat = len(self.W.data[0])

            self.W.grad = [[0.0]*out_feat for _ in range(in_feat)]
            self.b.grad = [[0.0]*out_feat]
            x.grad = [[0.0]*in_feat for _ in range(B)]

            for i in range(B):
                for j in range(out_feat):
                    g = out.grad[i][j]
                    self.b.grad[0][j] += g
                    for k in range(in_feat):
                        self.W.grad[k][j] += x.data[i][k] * g
                        x.grad[i][k] += self.W.data[k][j] * g

        out._backward = _backward
        out._prev = [x, self.W, self.b]
        return out

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def __call__(self, x):
        out_data = cpp_backend.relu_forward(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            def relu_grad(x_data, grad):
                if isinstance(x_data, list):
                    return [relu_grad(x_data[i], grad[i]) for i in range(len(x_data))]
                return grad if x_data > 0 else 0.0

            x.grad = relu_grad(x.data, out.grad)

        out._backward = _backward
        out._prev = [x]
        return out


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.W = Tensor(
            [[[[random.uniform(-0.1, 0.1) for _ in range(kernel_size)]
               for _ in range(kernel_size)]
              for _ in range(in_channels)]
             for _ in range(out_channels)],
            requires_grad=True
        )
        self.b = Tensor([0.0 for _ in range(out_channels)], requires_grad=True)
        self.kernel_size = kernel_size

    def __call__(self, x):
        out_data = cpp_backend.conv2d_forward(x.data, self.W.data, self.b.data)
        return Tensor(out_data, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]


class MaxPool2D:
    def __init__(self, kernel_size=2):
        self.k = kernel_size

    def __call__(self, x):
        out_data = cpp_backend.maxpool2d_forward(x.data, self.k)
        return Tensor(out_data, requires_grad=True)


class Flatten:
    def __call__(self, x):
        out_data = cpp_backend.flatten_forward(x.data)
        return Tensor(out_data, requires_grad=True)
