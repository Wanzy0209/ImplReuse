```python
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import os
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
        # Conversion: PyTorch uses CHW (Channels, Height, Width), TensorFlow uses HWC.
        # The original code permuted to (2, 0, 1). We keep HWC for TensorFlow standard.
        return tf.convert_to_tensor(image)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> tf.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        # Conversion: torch.from_numpy -> tf.convert_to_tensor
        # Conversion: .long() -> tf.cast(..., tf.int64)
        return tf.cast(tf.convert_to_tensor(mask), tf.int64)
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


class InfiniteSampler:
    
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
    # We use from_generator to wrap the existing Python logic (Dataset + Sampler)
    # Note: For high performance, native tf.data pipelines (from_tensor_slices + map) are preferred over from_generator.
    
    def generator():
        # Mimic the DataLoader loop using the Sampler
        for idx in sampler:
            yield dataset[idx]

    # Define the output signature for the TensorFlow dataset
    # Shapes are None because images might have different sizes
    output_signature = {
        "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    }

    tf_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    # Conversion: batch_size=12 -> .batch(12)
    # Conversion: pin_memory=True, persistent_workers=True -> .prefetch(tf.data.AUTOTUNE)
    # Conversion: itertools.cycle -> .repeat()
    tf_dataset = tf_dataset.batch(12)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    tf_dataset = tf_dataset.repeat()

    for step, batch in enumerate(tf_dataset.take(300)):
        # batch is a dict of tf.Tensors
        pass
```