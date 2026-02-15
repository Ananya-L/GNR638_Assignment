import math
from .tensor import Tensor

class CrossEntropyLoss:
    def __call__(self, logits, targets):

        B = len(targets)
        C = len(logits.data[0])

        probs = []
        loss = 0.0

        for i in range(B):
            row = logits.data[i]

            # Numerical stability trick
            max_logit = max(row)
            exps = [math.exp(v - max_logit) for v in row]
            s = sum(exps)
            p = [e / s for e in exps]

            probs.append(p)
            loss -= math.log(p[targets[i]])

        loss /= B
        out = Tensor(loss, requires_grad=True)

        def _backward():
            logits.grad = [[0.0]*C for _ in range(B)]
            for i in range(B):
                for j in range(C):
                    logits.grad[i][j] = probs[i][j]
                logits.grad[i][targets[i]] -= 1
                for j in range(C):
                    logits.grad[i][j] /= B

        out._backward = _backward
        out._prev = {logits}
        return out
