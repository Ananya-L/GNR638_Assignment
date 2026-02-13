import os
import time
import cv2
from framework.tensor import Tensor


def load_dataset(root_dir, image_size=(32, 32)):
    print("Loading dataset from:", root_dir)
    start = time.time()

    X = []
    y = []
    class_map = {}

    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_map[class_name] = idx

        for fname in os.listdir(class_path):
            if not fname.lower().endswith(".png"):
                continue

            img_path = os.path.join(class_path, fname)
            img = cv2.imread(img_path)

            img = cv2.resize(img, image_size)
            img = img.astype("float32") / 255.0

            # Convert HWC → CHW
            img = img.transpose(2, 0, 1)

            X.append(img.tolist())
            y.append(idx)

    elapsed = time.time() - start
    print(f"Dataset loading time: {elapsed:.2f} seconds")
    print(f"Class mapping: {class_map}")

    return Tensor(X, requires_grad=False), y