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
        # Conversion: PyTorch uses CHW (Channels, Height, Width), 
        # TensorFlow uses HWC (Height, Width, Channels) by default.
        # We return HWC for TensorFlow compatibility.
        return tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: torch.long is equivalent to tf.int64
        return tf.cast(tf.convert_to_tensor(mask), dtype=tf.int64)
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
        
        return {"image": image, "mask": mask.long()}
    
    def __len__(self):
        return len(self.image_paths)


# Conversion: InfiniteSampler is replaced by tf.data.Dataset.repeat() and .shuffle()
# in the main pipeline below.

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    
    # Conversion: PyTorch DataLoader + InfiniteSampler -> tf.data.Dataset
    # Create a dataset from the file paths
    image_paths = [str(p) for p in dataset.image_paths]
    mask_paths = [str(p) for p in dataset.mask_paths]
    
    # Create dataset from slices
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Conversion: InfiniteSampler (shuffle + infinite loop)
    # buffer_size=len ensures full shuffling equivalent to the sampler
    ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    ds = ds.repeat() # Infinite loop
    
    # Define the loading function (equivalent to __getitem__)
    def load_data(img_path, msk_path):
        # Decode bytes to string for cv2
        img_path = img_path.numpy().decode('utf-8')
        msk_path = msk_path.numpy().decode('utf-8')
        
        image = read_image(img_path, as_tensor=True)
        mask = read_mask(msk_path, as_tensor=True)
        
        return {"image": image, "mask": mask}
    
    # Output signature for the dict structure
    # Note: Image shape is HWC for TensorFlow
    output_signature = {
        "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    }
    
    # Map the loading function
    # Conversion: num_workers -> num_parallel_calls
    ds = ds.map(
        lambda x, y: tf.py_function(load_data, [x, y], output_signature),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Conversion: batch_size
    ds = ds.batch(12)
    
    # Conversion: pin_memory=True, persistent_workers=True -> prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # Conversion: itertools.cycle(dataloader) -> ds is already infinite
    for step, batch in enumerate(ds):
        if step >= 300:
            break
        # batch is a dict with keys 'image' and 'mask'
        pass
```