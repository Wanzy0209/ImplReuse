```python
import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path
import random
import itertools

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
        # Note: PyTorch uses CHW (Channels, Height, Width), TensorFlow uses HWC.
        # The source code permutes to CHW. We keep HWC for TensorFlow standard.
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
        
        # Conversion: .long() -> tf.cast(..., tf.int64)
        return {"image": image, "mask": tf.cast(mask, tf.int64)}
    
    def __len__(self):
        return len(self.image_paths)


class InfiniteSampler:
    # In TensorFlow, infinite sampling is typically handled by .repeat() on a tf.data.Dataset.
    # We keep this class to preserve structure, but it will be used to drive a generator.
    
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
    # We use from_generator to bridge the Python-based dataset/sampler logic to tf.data
    def gen():
        for idx in sampler:
            yield dataset[idx]

    # Define output shapes and types for the generator
    # Assuming images are HWC and masks are HW (grayscale)
    output_signature = {
        "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    }

    tf_dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # Conversion: DataLoader arguments -> tf.data operations
    # batch_size=12 -> .batch(12)
    # num_workers=8 -> .prefetch(tf.data.AUTOTUNE) (TF handles parallelism internally)
    # pin_memory=True -> Not needed in TF
    # persistent_workers=True -> Not needed in TF
    # InfiniteSampler -> .repeat() is implicit in the generator, but we can also call it.
    # However, since the generator is infinite, we don't strictly need .repeat() here.
    
    dataloader = tf_dataset.batch(12).prefetch(tf.data.AUTOTUNE)

    # Conversion: itertools.cycle(dataloader) -> The dataset is already infinite via the generator
    # We create an iterator to mimic the behavior of next(infinite_dataloader)
    infinite_dataloader = iter(dataloader)
    
    for step in range(300):
        batch = next(infinite_dataloader)
```