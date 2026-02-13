class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            for i in range(len(p.data)):
                if isinstance(p.data[i], list):
                    for j in range(len(p.data[i])):
                        p.data[i][j] -= self.lr * p.grad[i][j]
                else:
                    p.data[i] -= self.lr * p.grad[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = None