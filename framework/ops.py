from .tensor import Tensor

def add(a, b):
    # Handle list addition
    if isinstance(a.data, list):
        out_data = _add_lists(a.data, b.data)
    else:
        out_data = a.data + b.data
    
    out = Tensor(out_data, requires_grad=True)

    def _backward():
        if a.requires_grad:
            if a.grad is None:
                a.grad = out.grad
            else:
                a.grad = _add_lists(a.grad, out.grad) if isinstance(a.grad, list) else a.grad + out.grad
        if b.requires_grad:
            if b.grad is None:
                b.grad = out.grad
            else:
                b.grad = _add_lists(b.grad, out.grad) if isinstance(b.grad, list) else b.grad + out.grad

    out._backward = _backward
    out._prev = [a, b]
    return out


def mul(a, b):
    # Handle list multiplication
    if isinstance(a.data, list):
        out_data = _mul_lists(a.data, b.data)
    else:
        out_data = a.data * b.data
    
    out = Tensor(out_data, requires_grad=True)

    def _backward():
        if a.requires_grad:
            grad_a = _mul_scalar(b.data, out.grad)
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad = _add_lists(a.grad, grad_a) if isinstance(a.grad, list) else a.grad + grad_a
        if b.requires_grad:
            grad_b = _mul_scalar(a.data, out.grad)
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad = _add_lists(b.grad, grad_b) if isinstance(b.grad, list) else b.grad + grad_b

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
            grad_a = [
                [
                    sum(out.grad[i][j] * b.data[t][j] for j in range(len(b.data[0])))
                    for t in range(len(b.data))
                ]
                for i in range(len(a.data))
            ]
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad = _add_lists(a.grad, grad_a)

        if b.requires_grad:
            grad_b = [
                [
                    sum(a.data[i][t] * out.grad[i][j] for i in range(len(a.data)))
                    for j in range(len(b.data[0]))
                ]
                for t in range(len(b.data))
            ]
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad = _add_lists(b.grad, grad_b)

    out._backward = _backward
    out._prev = [a, b]
    return out


# Helper functions
def _add_lists(a, b):
    if isinstance(a, list):
        return [_add_lists(x, y) for x, y in zip(a, b)]
    else:
        return a + b


def _mul_lists(a, b):
    if isinstance(a, list):
        return [_mul_lists(x, y) for x, y in zip(a, b)]
    else:
        return a * b


def _mul_scalar(a, b):
    if isinstance(a, list):
        if isinstance(b, list):
            return [_mul_scalar(x, y) for x, y in zip(a, b)]
        else:
            return [_mul_scalar(x, b) for x in a]
    else:
        return a * b