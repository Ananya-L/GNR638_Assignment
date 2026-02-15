import argparse
import pickle
import json
from framework.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from data.loader import ImageDataset


def load_model(params, filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    loaded_params = model_data['parameters']
    for p, loaded_p in zip(params, loaded_params):
        p.data = loaded_p
    print(f"Model loaded from {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--weights_path', type=str, default='model.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except:
        config = {'conv_channels': 16, 'conv_kernel': 3}

    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    
    dataset = ImageDataset(args.dataset_path)
    num_classes = len(dataset.class_map)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {num_classes}")

    print("\n" + "="*60)
    print("MODEL")
    print("="*60)

    # Build model
    conv_ch = config.get('conv_channels', 16)
    conv_k = config.get('conv_kernel', 3)
    flat_size = conv_ch * 15 * 15
    
    conv = Conv2D(3, conv_ch, conv_k)
    relu = ReLU()
    pool = MaxPool2D(2)
    flat = Flatten()
    fc = Linear(flat_size, num_classes)

    # Load weights
    params = conv.parameters() + fc.parameters()
    load_model(params, args.weights_path)

    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    correct = 0
    total = 0
    batch_count = 0

    for X_batch, y_batch in dataset.get_batches(args.batch_size):
        # Forward pass
        out = conv(X_batch)
        out = relu(out)
        out = pool(out)
        out = flat(out)
        out = fc(out)

        # Predictions
        for i in range(len(y_batch)):
            if i >= len(out.data):
                break
            pred = out.data[i].index(max(out.data[i]))
            if pred == y_batch[i]:
                correct += 1
        
        total += len(y_batch)
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"Evaluated {total}/{len(dataset)} samples...")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == '__main__':
    main()