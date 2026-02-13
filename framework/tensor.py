class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = []

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # Initialize gradient if not provided
        if grad is None:
            if isinstance(self.data, list):
                grad = self._ones_like(self.data)
            else:
                grad = 1.0

        # Accumulate gradient (important fix!)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self._add_grad(self.grad, grad)

        # Call backward function
        self._backward()

        # Propagate to dependencies
        for t in self._prev:
            if hasattr(t, '_backward'):
                t._backward()

    def _ones_like(self, data):
        if isinstance(data, list):
            return [self._ones_like(d) for d in data]
        else:
            return 1.0

    def _add_grad(self, g1, g2):
        if isinstance(g1, list):
            return [self._add_grad(a, b) for a, b in zip(g1, g2)]
        else:
            return g1 + g2

    def zero_grad(self):
        self.grad = None