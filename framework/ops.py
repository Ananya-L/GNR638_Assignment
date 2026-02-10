from .tensor import Tensor

def add(a, b):
    out = Tensor(a.data + b.data, requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = out.grad
        if b.requires_grad:
            b.grad = out.grad

    out._backward = _backward
    out._prev = [a, b]
    return out


def mul(a, b):
    out = Tensor(a.data * b.data, requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = b.data * out.grad
        if b.requires_grad:
            b.grad = a.data * out.grad

    out._backward = _backward
    out._prev = [a, b]
    return out

def matmul(a, b):
    # a: [m x n], b: [n x k]
    out_data = [
        [
            sum(a.data[i][t] * b.data[t][j] for t in range(len(b.data)))
            for j in range(len(b.data[0]))
        ]
        for i in range(len(a.data))
    ]

    out = Tensor(out_data, requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = [
                [
                    sum(out.grad[i][j] * b.data[t][j] for j in range(len(b.data[0])))
                    for t in range(len(b.data))
                ]
                for i in range(len(a.data))
            ]

        if b.requires_grad:
            b.grad = [
                [
                    sum(a.data[i][t] * out.grad[i][j] for i in range(len(a.data)))
                    for j in range(len(b.data[0]))
                ]
                for t in range(len(b.data))
            ]

    out._backward = _backward
    out._prev = [a, b]
    return out
