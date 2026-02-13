import argparse
import pickle
import json
from framework.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from framework.tensor import Tensor
from data.loader import load_dataset


def load_model(params, filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    loaded_params = model_data['parameters']
    for p, loaded_p in zip(params, loaded_params):
        p.data = loaded_p
    print(f"Model loaded from {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dataset')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--weights_path', type=str, default='model.pkl')
    
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
    X_test, y_test = load_dataset(args.dataset_path)
    num_classes = max(y_test) + 1
    num_samples = len(y_test)
    print(f"Number of samples: {num_samples}")
    print(f"Number of classes detected: {num_classes}")
    
    # Force minimum 2 classes to match training
    if num_classes < 2:
        print("WARNING: Only 1 class detected. Using 2 classes to match training.")
        num_classes = 2

    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)

    # Build model (must match training exactly)
    conv_ch = config.get('conv_channels', 16)
    conv_k = config.get('conv_kernel', 3)
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

    # Load weights
    params = conv.parameters() + fc.parameters()
    load_model(params, args.weights_path)

    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    correct = 0
    total = len(y_test)

    for i in range(total):
        x = Tensor([X_test.data[i]], requires_grad=False)

        out = conv(x)
        out = relu(out)
        out = pool(out)
        out = flat(out)
        out = fc(out)

        pred = out.data[0].index(max(out.data[0]))

        if pred == y_test[i]:
            correct += 1
        
        if (i + 1) % 500 == 0:
            print(f"Evaluated {i+1}/{total} samples...")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinal Evaluation Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == '__main__':
    main()