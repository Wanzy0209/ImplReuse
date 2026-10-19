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
    # Conversion: Read image using cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        # Conversion: PyTorch uses CHW (Channels, Height, Width), TensorFlow uses HWC.
        # We preserve HWC here as it is the TensorFlow standard.
        return tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        mask = tf.convert_to_tensor(mask, dtype=tf.int8)
    return mask

class SegDataset:
    """Maks dataset for segmentation."""
    
    def __init__(
        self, 
        image_dir: str | Path,
        mask_dir: str | Path, 
    ):
        # super().__init__() # Not needed for plain Python class
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = list(self.image_dir.glob("*"))
        # Conversion: Ensure paths are strings for TensorFlow compatibility
        self.image_paths = [str(i) for i in image_paths if is_image_file(i)]

        self.mask_paths = []
        for img_p in self.image_paths:
            file_name = os.path.splitext(os.path.basename(img_p))[0]
            self.mask_paths.append(os.path.join(self.mask_dir, file_name + ".png"))

# Conversion: InfiniteSampler is replaced by dataset.shuffle().repeat()
# Conversion: DataLoader is replaced by tf.data.Dataset operations

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    
    # Conversion: Create a tf.data.Dataset from the file paths
    # This replaces the need for a custom Sampler class
    paths_dataset = tf.data.Dataset.from_tensor_slices((dataset.image_paths, dataset.mask_paths))
    
    # Conversion: InfiniteSampler logic (shuffle + infinite repeat)
    # buffer_size=len(dataset) ensures full shuffling like the PyTorch sampler
    paths_dataset = paths_dataset.shuffle(buffer_size=len(dataset.image_paths), reshuffle_each_iteration=True)
    paths_dataset = paths_dataset.repeat() # Makes it infinite
    
    def load_data(image_path, mask_path):
        # Using tf.py_function to wrap the custom cv2 reading logic
        # This preserves the exact image processing steps from the source
        image, mask = tf.py_function(
            func=lambda p1, p2: (read_image(p1, as_tensor=True), read_mask(p2, as_tensor=True)),
            inp=[image_path, mask_path],
            Tout=[tf.float32, tf.int8]
        )
        
        # Conversion: mask.long() in PyTorch is int64
        mask = tf.cast(mask, tf.int64)
        
        return {"image": image, "mask": mask}

    # Conversion: DataLoader configuration
    # num_workers -> num_parallel_calls
    # pin_memory -> handled automatically by TF
    # persistent_workers -> handled automatically by TF
    dataloader = paths_dataset.map(
        load_data, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataloader = dataloader.batch(12)
    dataloader = dataloader.prefetch(tf.data.AUTOTUNE)
    
    # Conversion: itertools.cycle is not needed because .repeat() makes the dataset infinite
    for step, batch in enumerate(dataloader):
        if step >= 300:
            break
        # batch is a dict of tensors
```