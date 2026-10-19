```python
import tensorflow as tf
import cv2
import numpy as np
import os
import itertools
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_image_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS

def read_image(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    # Conversion: cv2 reads images as HWC, PyTorch typically uses CHW.
    # TensorFlow uses HWC by default, so we omit the permute(2, 0, 1) step.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        return tf.convert_to_tensor(image)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        # Conversion: .long() -> tf.int64
        return tf.cast(mask, tf.int64)
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
    
    # Conversion: PyTorch __getitem__ is replaced by a mapping function in tf.data pipeline
    def to_tf_dataset(self):
        # Convert paths to strings for TensorFlow
        img_paths = [str(p) for p in self.image_paths]
        mask_paths = [str(p) for p in self.mask_paths]
        
        # Create a dataset of file paths
        ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        
        # Define the loading logic to be used in map
        def load_data(img_path, mask_path):
            # Use tf.py_function to wrap cv2 operations (non-TF ops)
            img_path = img_path.numpy().decode('utf-8')
            mask_path = mask_path.numpy().decode('utf-8')
            
            image = read_image(img_path, as_tensor=True)
            mask = read_mask(mask_path, as_tensor=True)
            
            # Return tuple (image, mask) instead of dict for standard TF pipeline
            return image, mask

        # Map the loading function
        # Conversion: num_workers -> num_parallel_calls
        ds = ds.map(
            lambda x, y: tf.py_function(load_data, [x, y], [tf.float32, tf.int64]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return ds

# Conversion: InfiniteSampler is replaced by .repeat() in the tf.data pipeline
# class InfiniteSampler(Sampler): ...

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    
    # Conversion: DataLoader setup -> tf.data pipeline
    # InfiniteSampler logic is handled by .repeat()
    # shuffle=True is handled by .shuffle()
    # batch_size is handled by .batch()
    # num_workers/pin_memory/persistent_workers are handled by .prefetch() and AUTOTUNE
    
    ds = dataset.to_tf_dataset()
    
    # Shuffle buffer size should ideally be the size of the dataset for perfect shuffling
    # or a sufficiently large number.
    ds = ds.shuffle(buffer_size=1000) 
    ds = ds.batch(12)
    ds = ds.repeat() # Replaces InfiniteSampler
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Replaces num_workers/pin_memory optimization
    
    # Conversion: itertools.cycle(dataloader) -> iter(ds) (since ds is already repeating)
    infinite_dataloader = iter(ds)
    
    for step in range(300):
        # Conversion: batch is now a tuple (images, masks) instead of a dict
        batch = next(infinite_dataloader)
```