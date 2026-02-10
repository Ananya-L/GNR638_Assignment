class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = []

    def backward(self, grad=1.0):
        if not self.requires_grad:
            return

        self.grad = grad
        self._backward()


        for t in self._prev:
            t._backward()
