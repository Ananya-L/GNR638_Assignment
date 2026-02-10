from framework.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from framework.loss import CrossEntropyLoss
from framework.optim import SGD
from framework.tensor import Tensor
from data.loader import load_dataset


# ---------------- Parameter counting ----------------
def count_params(params):
    def count(x):
        if isinstance(x, list):
            return sum(count(v) for v in x)
        return 1

    total = 0
    for p in params:
        total += count(p.data)
    return total


# ---------------- Load dataset ----------------
X, y = load_dataset("dataset")   # <-- CHANGE THIS PATH


# ---------------- Model definition ----------------
conv = Conv2D(3, 2, 3)     # in_channels=3, out_channels=2, kernel=3
relu = ReLU()
pool = MaxPool2D(2)
flat = Flatten()
fc = Linear(2, 2)          # minimal FC layer (allowed)


# ---------------- Collect parameters ----------------
params = conv.parameters() + fc.parameters()

total_params = count_params(params)
print("Total trainable parameters:", total_params)


# ---------------- MACs & FLOPs calculation ----------------
# Input: 1 x 32 x 32
# Conv output: 2 x 30 x 30 (no padding, stride=1)

C_in = 3
C_out = 2
K = 3
H_out = 30
W_out = 30

conv_macs = C_out * H_out * W_out * C_in * K * K
fc_macs = 2 * 2   # Linear(2 -> 2)

total_macs = conv_macs + fc_macs
total_flops = 2 * total_macs

print("MACs per forward pass:", total_macs)
print("FLOPs per forward pass:", total_flops)


# ---------------- Optimizer & loss ----------------
opt = SGD(params, lr=0.01)
loss_fn = CrossEntropyLoss()


# ---------------- Training loop ----------------
for epoch in range(5):
    out = conv(X)
    out = relu(out)
    out = pool(out)
    out = flat(out)
    out = fc(out)

    loss = loss_fn(out, y)

    loss.backward()
    opt.step()
    opt.zero_grad()

    print("Epoch", epoch, "Loss:", loss.data)
