import argparse
import pickle
import json
import time
from framework.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from framework.loss import CrossEntropyLoss
from framework.optim import SGD
from data.loader import ImageDataset


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
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--save_path', type=str, default='model.pkl')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except:
        config = {'conv_channels': 16, 'conv_kernel': 3}

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = ImageDataset(args.dataset_path)
    num_classes = len(dataset.class_map)
    num_samples = len(dataset)

    print(f"Total samples: {num_samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {args.batch_size}")

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    conv_ch = config.get('conv_channels', 16)
    conv_k = config.get('conv_kernel', 3)

    # After Conv: 32 - k + 1
    conv_output = 32 - conv_k + 1

    # After MaxPool 2x2
    pooled_output = conv_output // 2

    flat_size = conv_ch * pooled_output * pooled_output

    conv = Conv2D(3, conv_ch, conv_k)
    relu = ReLU()
    pool = MaxPool2D(2)
    flat = Flatten()
    fc = Linear(flat_size, num_classes)

    print(f"Conv2D: 3 -> {conv_ch} channels, {conv_k}x{conv_k} kernel")
    print("ReLU activation")
    print("MaxPool2D: 2x2 kernel")
    print(f"Flatten: {conv_ch}x{pooled_output}x{pooled_output} -> {flat_size}")
    print(f"Linear: {flat_size} -> {num_classes}")

    params = conv.parameters() + fc.parameters()
    total_params = count_params(params)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # MACs & FLOPs
    conv_macs = conv_ch * conv_output * conv_output * 3 * conv_k * conv_k
    fc_macs = flat_size * num_classes
    total_macs = conv_macs + fc_macs
    total_flops = 2 * total_macs

    print(f"MACs per forward pass: {total_macs:,}")
    print(f"FLOPs per forward pass: {total_flops:,}")

    opt = SGD(params, lr=args.lr)
    loss_fn = CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        epoch_start = time.time()

        for X_batch, y_batch in dataset.get_batches(args.batch_size):

            if X_batch is None or len(y_batch) == 0:
                continue

            # Forward pass
            out = conv(X_batch)
            out = relu(out)
            out = pool(out)
            out = flat(out)
            out = fc(out)

            # Loss
            loss = loss_fn(out, y_batch)

            # Backward
            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_loss += loss.data
            batch_size_actual = len(y_batch)

            # Accuracy
            for i in range(batch_size_actual):
                pred = out.data[i].index(max(out.data[i]))
                if pred == y_batch[i]:
                    epoch_correct += 1

            epoch_total += batch_size_actual
            batch_count += 1

            if batch_count % 50 == 0:
                print(f"  Batch {batch_count} | Samples: {epoch_total}/{num_samples}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / batch_count
        acc = epoch_correct / epoch_total

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {acc:.4f} | "
              f"Time: {epoch_time:.2f}s")

    print("\n" + "=" * 60)
    save_model(params, args.save_path)
    print("Training complete!")


if __name__ == '__main__':
    main()
