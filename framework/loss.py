import math
from .tensor import Tensor

class CrossEntropyLoss:
    def __call__(self, logits, targets):

        # Ensure logits are [B x C]
        if isinstance(logits.data[0], list):
            logits_data = logits.data
        else:
            logits_data = [logits.data]

        B = min(len(logits_data), len(targets))
        logits_data = logits_data[:B]
        C = len(logits_data[0])


        probs = []
        loss = 0.0

        for i in range(B):
            exps = [math.exp(v) for v in logits_data[i]]
            s = sum(exps)
            p = [e / s for e in exps]
            probs.append(p)

            assert targets[i] >= 0 and targets[i] < C, \
                f"Target {targets[i]} out of range for {C} classes"

            loss -= math.log(p[targets[i]])

        loss /= B
        out = Tensor(loss, requires_grad=True)

        def _backward():
            logits.grad = [[0.0] * C for _ in range(B)]
            for i in range(B):
                for j in range(C):
                    logits.grad[i][j] = probs[i][j]
                logits.grad[i][targets[i]] -= 1
            for i in range(B):
                for j in range(C):
                    logits.grad[i][j] /= B

        out._backward = _backward
        out._prev = [logits]
        return out