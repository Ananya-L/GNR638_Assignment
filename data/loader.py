import os
import cv2
from framework.tensor import Tensor


class ImageDataset:
    def __init__(self, root_dir, image_size=(32, 32)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []
        self.class_map = {}

        # Scan directory structure
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            self.class_map[class_name] = idx

            for fname in os.listdir(class_path):
                if fname.lower().endswith('.png'):
                    img_path = os.path.join(class_path, fname)
                    self.samples.append((img_path, idx))

        print(f"Found {len(self.samples)} images in {len(self.class_map)} classes")
        print(f"Class mapping: {self.class_map}")

    def __len__(self):
        return len(self.samples)

    def load_batch(self, indices):
        """Load specific batch of images (STRICT — no silent skipping)"""
        X_batch = []
        y_batch = []

        for idx in indices:
            img_path, label = self.samples[idx]

            img = cv2.imread(img_path)
            if img is None:
                # STOP immediately if image fails
                raise ValueError(f"Failed to load image: {img_path}")

            img = cv2.resize(img, self.image_size)
            img = img.astype("float32") / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            X_batch.append(img.tolist())
            y_batch.append(label)

        # Always return consistent batch
        return Tensor(X_batch, requires_grad=False), y_batch

    def get_batches(self, batch_size=32):
        """Generator that yields STRICT fixed-size batches"""
        num_samples = len(self.samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size

            # Skip last incomplete batch
            if end_idx > num_samples:
                break

            indices = list(range(start_idx, end_idx))
            X_batch, y_batch = self.load_batch(indices)

            yield X_batch, y_batch
