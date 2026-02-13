import argparse
import pickle
import json
from framework.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from framework.loss import CrossEntropyLoss
from framework.optim import SGD
from framework.tensor import Tensor
from data.loader import load_dataset


def count_params(params):
    def count(x):
        if isinstance(x, list):
            return sum(count(v) for v in x)
        return 1

    total = 0
    for p in params:
        total += count(p.data)
    return total


def save_model(params, filepath):
    model_data = {'parameters': [p.data for p in params]}
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dataset')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--save_path', type=str, default='model.pkl')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except:
        config = {'conv_channels': 16, 'conv_kernel': 3}

    print("="*60)
    print("DATASET LOADING")
    print("="*60)
    
    # Load dataset
    X, y = load_dataset(args.dataset_path)
    num_classes = max(y) + 1  # This will be 1 for your dataset
    num_samples = len(y)
    print(f"Number of samples: {num_samples}")
    print(f"Number of classes detected: {num_classes}")
    
    # Force minimum 2 classes for CrossEntropy
    if num_classes < 2:
        print("WARNING: Only 1 class detected. Using 2 classes for training.")
        num_classes = 2

    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    
    # Build model
    conv_ch = config.get('conv_channels', 16)
    conv_k = config.get('conv_kernel', 3)
    
    # Calculate flattened size
    # Input: 32x32x3 (RGB)
    # After Conv (k=3, stride=1, no padding): 30x30xconv_ch
    # After MaxPool (k=2, stride=2): 15x15xconv_ch
    flat_size = conv_ch * 15 * 15
    
    conv = Conv2D(3, conv_ch, conv_k)
    relu = ReLU()
    pool = MaxPool2D(2)
    flat = Flatten()
    fc = Linear(flat_size, num_classes)

    print(f"Conv2D: 3 -> {conv_ch} channels, {conv_k}x{conv_k} kernel")
    print(f"ReLU activation")
    print(f"MaxPool2D: 2x2 kernel")
    print(f"Flatten: {conv_ch}x15x15 -> {flat_size} features")
    print(f"Linear: {flat_size} -> {num_classes} classes")

    # Collect parameters
    params = conv.parameters() + fc.parameters()
    total_params = count_params(params)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Calculate MACs & FLOPs
    H_out = 32 - conv_k + 1  # 30 for k=3
    W_out = 32 - conv_k + 1
    
    # Conv MACs: output_channels * output_H * output_W * input_channels * kernel_H * kernel_W
    conv_macs = conv_ch * H_out * W_out * 3 * conv_k * conv_k
    
    # FC MACs: input_features * output_features
    fc_macs = flat_size * num_classes
    
    total_macs = conv_macs + fc_macs
    total_flops = 2 * total_macs

    print(f"MACs per forward pass: {total_macs:,}")
    print(f"FLOPs per forward pass: {total_flops:,}")

    # Optimizer & loss
    opt = SGD(params, lr=args.lr)
    loss_fn = CrossEntropyLoss()

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(args.epochs):
        out = conv(X)
        out = relu(out)
        out = pool(out)
        out = flat(out)
        out = fc(out)

        loss = loss_fn(out, y)

        loss.backward()
        opt.step()
        opt.zero_grad()

        # Calculate accuracy
        correct = 0
        for i in range(len(y)):
            pred = out.data[i].index(max(out.data[i]))
            if pred == y[i]:
                correct += 1
        acc = correct / len(y)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.data:.4f} | Accuracy: {acc:.4f}")

    # Save model
    print("\n" + "="*60)
    save_model(params, args.save_path)
    print("Training complete!")


if __name__ == '__main__':
    main()