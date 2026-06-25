```python
import tensorflow as tf
import cv2
import numpy as np
import itertools
import random
import gc
import os
from pathlib import Path

# Conversion: Removed torch imports, added tensorflow and os imports

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
        # Conversion: PyTorch uses (C, H, W) format, TensorFlow uses (H, W, C).
        # Removed .permute(2, 0, 1) to maintain TensorFlow's default channel-last format.
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
        # Conversion: Removed super().__init__() as we are not inheriting from torch.utils.data.Dataset
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
    # Conversion: Removed inheritance from torch.utils.data.Sampler
    # This class is kept to preserve structure and logic, used in the generator below.
    
    def __init__(self, data_source: SegDataset, shuffle: bool = True):
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
    # We use from_generator to integrate the custom Python logic (OpenCV reading) 
    # and the InfiniteSampler logic.
    
    def data_generator():
        # Iterate using the InfiniteSampler logic
        for index in sampler:
            yield dataset[index]

    # Define the output signature for the TensorFlow dataset
    # Images are (H, W, 3) float32, Masks are (H, W) int64
    output_signature = {
        "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    }

    # Create the tf.data.Dataset
    # Conversion: DataLoader arguments mapped to tf.data operations
    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )
    
    # Conversion: batch_size=12 -> .batch(12)
    tf_dataset = tf_dataset.batch(12)
    
    # Conversion: num_workers=8, pin_memory=True, persistent_workers=True
    # In TensorFlow, performance is typically handled by prefetching and parallel mapping.
    # AUTOTUNE allows the runtime to tune the buffer size dynamically.
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    # Conversion: itertools.cycle(dataloader)
    # Since InfiniteSampler is already infinite, the dataset is infinite.
    # We create an iterator to fetch batches.
    infinite_dataloader = iter(tf_dataset)
    
    for step in range(300):
        # Conversion: next(infinite_dataloader)
        batch = next(infinite_dataloader)
```