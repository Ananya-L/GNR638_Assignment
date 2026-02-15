from .tensor import Tensor
from .ops import matmul, add
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
            [[0.0 for _ in range(out_features)]],
            requires_grad=True
        )

    def __call__(self, x):
        return add(matmul(x, self.W), self.b)

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def __call__(self, x):

        def relu_forward(data):
            if isinstance(data, list):
                return [relu_forward(v) for v in data]
            else:
                return max(0.0, data)

        out_data = relu_forward(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def relu_backward(x_data, out_grad):
            if isinstance(x_data, list):
                return [relu_backward(x_data[i], out_grad[i]) for i in range(len(x_data))]
            else:
                return out_grad if x_data > 0 else 0.0

        def _backward():
            x.grad = relu_backward(x.data, out.grad)

        out._backward = _backward
        out._prev = [x]
        return out

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.W = Tensor(
            [
                [
                    [
                        [random.uniform(-0.1, 0.1) for _ in range(kernel_size)]
                        for _ in range(kernel_size)
                    ]
                    for _ in range(in_channels)
                ]
                for _ in range(out_channels)
            ],
            requires_grad=True
        )

        self.b = Tensor([0.0 for _ in range(out_channels)], requires_grad=True)

    def __call__(self, x):
        # x: [B, C, H, W]
        B, C, H, W = len(x.data), len(x.data[0]), len(x.data[0][0]), len(x.data[0][0][0])
        k = self.kernel_size
        out_h = H - k + 1
        out_w = W - k + 1

        out_data = cpp_backend.conv2d_forward(
                x.data,
                self.W.data
            )

        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        s = 0.0
                        for ic in range(C):
                            for ki in range(k):
                                for kj in range(k):
                                    s += (
                                        x.data[b][ic][i+ki][j+kj] *
                                        self.W.data[oc][ic][ki][kj]
                                    )
                        out_data[b][oc][i][j] = s + self.b.data[oc]

        out = Tensor(out_data, requires_grad=True)

        def _backward():
            self.W.grad = [
                [
                    [[0.0]*k for _ in range(k)]
                    for _ in range(C)
                ]
                for _ in range(self.out_channels)
            ]
            self.b.grad = [0.0 for _ in range(self.out_channels)]
            x.grad = [
                [
                    [[0.0]*W for _ in range(H)]
                    for _ in range(C)
                ]
                for _ in range(B)
            ]

            for b in range(B):
                for oc in range(self.out_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            g = out.grad[b][oc][i][j]
                            self.b.grad[oc] += g
                            for ic in range(C):
                                for ki in range(k):
                                    for kj in range(k):
                                        self.W.grad[oc][ic][ki][kj] += \
                                            x.data[b][ic][i+ki][j+kj] * g
                                        x.grad[b][ic][i+ki][j+kj] += \
                                            self.W.data[oc][ic][ki][kj] * g

        out._backward = _backward
        out._prev = [x, self.W, self.b]
        return out

    def parameters(self):
        return [self.W, self.b]


class MaxPool2D:
    def __init__(self, kernel_size=2):
        self.k = kernel_size

    def __call__(self, x):
        B, C, H, W = len(x.data), len(x.data[0]), len(x.data[0][0]), len(x.data[0][0][0])
        k = self.k
        out_h = H // k
        out_w = W // k

        out_data = [
            [
                [
                    [0.0 for _ in range(out_w)]
                    for _ in range(out_h)
                ]
                for _ in range(C)
            ]
            for _ in range(B)
        ]

        self.max_idx = {}

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        block = []
                        for ki in range(k):
                            for kj in range(k):
                                block.append((x.data[b][c][i*k+ki][j*k+kj], ki, kj))
                        m, mi, mj = max(block)
                        out_data[b][c][i][j] = m
                        self.max_idx[(b, c, i, j)] = (mi, mj)

        out = Tensor(out_data, requires_grad=True)

        def _backward():
            x.grad = [
                [
                    [[0.0]*W for _ in range(H)]
                    for _ in range(C)
                ]
                for _ in range(B)
            ]
            for (b, c, i, j), (mi, mj) in self.max_idx.items():
                x.grad[b][c][i*k+mi][j*k+mj] += out.grad[b][c][i][j]

        out._backward = _backward
        out._prev = [x]
        return out

class Flatten:
    def __call__(self, x):
        B = len(x.data)
        flat = []
        for b in range(B):
            flat.append(
                [v for ch in x.data[b] for row in ch for v in row]
            )

        out = Tensor(flat, requires_grad=True)

        def _backward():
            x.grad = []
            for b in range(B):
                idx = 0
                restored = []
                for ch in x.data[b]:
                    channel = []
                    for row in ch:
                        channel.append(out.grad[b][idx:idx+len(row)])
                        idx += len(row)
                    restored.append(channel)
                x.grad.append(restored)

        out._backward = _backward
        out._prev = [x]
        return out