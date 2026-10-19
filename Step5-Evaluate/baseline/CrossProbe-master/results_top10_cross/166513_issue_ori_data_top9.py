```python
import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path
import random
import gc
import itertools

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_image_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS

def read_image(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    # Conversion: cv2 reads as BGR, convert to RGB
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        # Note: PyTorch code permuted to (C, H, W). TensorFlow standard is (H, W, C).
        # We keep (H, W, C) here for compatibility with TF layers.
        return tf.convert_to_tensor(image)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        return tf.convert_to_tensor(mask)
    return mask

class SegDataset:
    """Maks dataset for segmentation."""
    
    def __init__(
        self, 
        image_dir: str | Path,
        mask_dir: str | Path, 
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = list(self.image_dir.glob("*"))
        # Conversion: Store paths as strings for TensorFlow compatibility
        self.image_paths = [str(i) for i in image_paths if is_image_file(i)]

        self.mask_paths = []
        for img_p in self.image_paths:
            file_name = os.path.splitext(os.path.basename(img_p))[0]
            self.mask_paths.append(os.path.join(self.mask_dir, file_name + ".png"))
    
    def __getitem__(self, index):
        # Kept for structural compatibility, though TF pipeline uses map
        image_path = self.image_paths[index]
        image = read_image(image_path, as_tensor=True)

        mask_path = self.mask_paths[index]
        mask = read_mask(mask_path, as_tensor=True)
        
        # Conversion: mask.long() -> tf.cast(mask, tf.int64)
        return {"image": image, "mask": tf.cast(mask, tf.int64)}
    
    def __len__(self):
        return len(self.image_paths)

# InfiniteSampler is replaced by tf.data.Dataset.repeat() in the pipeline below.

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    
    # Conversion: DataLoader -> tf.data.Dataset pipeline
    # Create dataset from file paths
    img_ds = tf.data.Dataset.from_tensor_slices(dataset.image_paths)
    mask_ds = tf.data.Dataset.from_tensor_slices(dataset.mask_paths)
    ds = tf.data.Dataset.zip((img_ds, mask_ds))

    # Define the loading function (equivalent to __getitem__)
    def load_data(img_path, mask_path):
        # Use tf.py_function to wrap cv2 operations (non-TF ops)
        img, mask = tf.py_function(
            func=lambda p1, p2: (read_image(p1, True), read_mask(p2, True)),
            inp=[img_path, mask_path],
            Tout=[tf.float32, tf.int8]
        )
        
        # Set shapes (required for batching)
        img.set_shape([None, None, 3]) # HWC
        mask.set_shape([None, None])
        
        # Conversion: mask.long() -> tf.cast(mask, tf.int64)
        mask = tf.cast(mask, tf.int64)
        
        return {"image": img, "mask": mask}

    # Build the pipeline
    # num_workers -> num_parallel_calls
    # sampler (shuffle) -> shuffle
    # InfiniteSampler -> repeat
    # batch_size -> batch
    # pin_memory/persistent_workers -> prefetch
    ds = ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(dataset)) # Shuffle logic
    ds = ds.repeat() # Infinite iteration
    ds = ds.batch(12)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Conversion: itertools.cycle(dataloader) -> iter(ds)
    # Since ds is infinite, we just iterate directly.
    for step, batch in enumerate(ds):
        if step >= 300:
            break
        # batch is a dict of tensors
```