from framework.tensor import Tensor
from framework.ops import add
from framework.ops import matmul
from framework.layers import ReLU
a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)

c = add(a, b)
c.backward()

print(a.grad, b.grad)  # should print: 1.0 1.0

from framework.ops import mul

a = Tensor(2.0, requires_grad=True)
b = Tensor(4.0, requires_grad=True)

c = mul(a, b)
c.backward()

print(a.grad, b.grad)  # expected: 4.0 2.0

a = Tensor([[1, 2]], requires_grad=True)   # 1×2
b = Tensor([[3], [4]], requires_grad=True) # 2×1

c = matmul(a, b)
c.backward([[1]])

print(a.grad)  # [[3, 4]]
print(b.grad)  # [[1], [2]]

x = Tensor([[1.0, -2.0]], requires_grad=True)
relu = ReLU()

y = relu(x)
y.backward([[1.0, 1.0]])

print(x.grad)  # [[1.0, 0.0]]

from framework.loss import CrossEntropyLoss

logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
targets = [0]

loss_fn = CrossEntropyLoss()
loss = loss_fn(logits, targets)
loss.backward()

print(loss.data)
print(logits.grad)