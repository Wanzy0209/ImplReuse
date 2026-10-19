```python
import tensorflow as tf
import cv2
import numpy as np
import itertools
import random
import gc
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
        # Conversion: PyTorch uses CHW (Channels, Height, Width) via permute(2, 0, 1).
        # TensorFlow uses HWC (Height, Width, Channels) by default.
        return tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: PyTorch .long() maps to tf.int64
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
        
        return {"image": image, "mask": mask}
    
    def __len__(self):
        return len(self.image_paths)

# Conversion: InfiniteSampler is replaced by .repeat() in the tf.data pipeline.
# Custom Sampler classes are not needed in TensorFlow.

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    
    # Conversion: Create a tf.data.Dataset from indices to utilize the SegDataset class
    indices = tf.range(len(dataset), dtype=tf.int32)
    
    def load_item(index):
        # Wrapper to call the dataset's __getitem__
        # Note: __getitem__ returns a dict of tensors
        return dataset[index]

    # Conversion: DataLoader setup
    # 1. Create dataset from indices
    ds = tf.data.Dataset.from_tensor_slices(indices)
    
    # 2. Shuffle (mimics InfiniteSampler shuffle=True)
    # Buffer size determines how much shuffling occurs. 
    ds = ds.shuffle(buffer_size=len(dataset))
    
    # 3. Map the loading function (mimics num_workers via parallel calls)
    # We use tf.py_function to ensure the cv2 operations (which are numpy based) 
    # run correctly inside the graph, especially with dynamic shapes.
    output_signature = {
        "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    }
    ds = ds.map(
        lambda x: tf.py_function(load_item, [x], output_signature),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 4. Batch (mimics batch_size=12)
    ds = ds.batch(12)
    
    # 5. Repeat (mimics InfiniteSampler)
    ds = ds.repeat()
    
    # 6. Prefetch (mimics pin_memory=True and persistent_workers=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Iteration
    # Conversion: itertools.cycle is not needed as ds.repeat() makes the dataset infinite
    for step, batch in enumerate(ds):
        if step >= 300:
            break
        # batch is a dict of tensors
        pass
```