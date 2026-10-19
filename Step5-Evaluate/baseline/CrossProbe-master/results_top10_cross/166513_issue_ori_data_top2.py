```python
import tensorflow as tf
import itertools
import random
import gc
import cv2
import numpy as np
from pathlib import Path
import os

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_image_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS

def read_image(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        # Note: PyTorch uses CHW (Channels, Height, Width), TensorFlow uses HWC (Height, Width, Channels).
        # The original code permutes to CHW. Here we keep HWC for TensorFlow standard.
        return tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        mask = tf.convert_to_tensor(mask)
    return mask

class SegDataset:
    """Maks dataset for segmentation."""
    
    def __init__(
        self, 
        image_dir: str | Path,
        mask_dir: str | Path, 
    ):
        # Conversion: Removed super().__init__() as we no longer inherit from torch.utils.data.Dataset
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = list(self.image_dir.glob("*"))
        self.image_paths = [i for i in image_paths if is_image_file(i)]

        self.mask_paths = []
        for img_p in self.image_paths:
            file_name = os.path.splitext(os.path.basename(img_p))[0]
            self.mask_paths.append(os.path.join(self.mask_dir, file_name + ".png"))
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = read_image(image_path, as_tensor=True)

        mask_path = self.mask_paths[index]
        mask = read_mask(mask_path, as_tensor=True)
        
        # Conversion: mask.long() -> tf.cast(mask, tf.int64)
        return {"image": image, "mask": tf.cast(mask, tf.int64)}
    
    def __len__(self):
        return len(self.image_paths)


class InfiniteSampler:
    # Conversion: torch.utils.data.Sampler -> Python Iterator
    # In TensorFlow, infinite sampling is often handled by .repeat(), 
    # but we preserve the class structure here to mimic the original logic.
    
    def __init__(self, data_source, shuffle: bool = True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.num_samples = len(self.data_source)
        self.start_index = 0

    def __iter__(self):
        while True:
            indices = list(range(self.num_samples))
            if self.shuffle:
                random.shuffle(indices)
            yield from indices

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    sampler = InfiniteSampler(dataset)
    
    # Conversion: torch.utils.data.DataLoader -> tf.data.Dataset
    # We use from_generator to bridge the Python-based dataset/sampler to TF
    def generator():
        for idx in sampler:
            yield dataset[idx]

    # Define output signature for the generator
    output_signature = {
        "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    }

    # Conversion: DataLoader initialization
    # batch_size=12 -> .batch(12)
    # num_workers=8 -> .prefetch(tf.data.AUTOTUNE) (approximate equivalent for pipeline efficiency)
    # pin_memory=True -> Handled automatically by TensorFlow
    # persistent_workers=True -> Handled automatically by tf.data pipeline
    tf_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    tf_dataset = tf_dataset.batch(12)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Conversion: itertools.cycle(dataloader)
    # The dataset is already infinite due to InfiniteSampler, so cycle is not strictly needed.
    # We iterate directly.
    for step, batch in enumerate(tf_dataset):
        if step >= 300:
            break
        # batch is a dict of tf.Tensors
        pass
```