```python
import tensorflow as tf
from pathlib import Path
import os

# Conversion: Removed torch, cv2, numpy, itertools, random, gc
# Using tensorflow for image operations and data pipeline

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_image_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS

# Conversion: Replaced cv2 and torch operations with tf.io and tf.image
# Removed as_tensor argument as TensorFlow always uses Tensors
def read_image(path: str | Path) -> tf.Tensor:
    # Read file
    image = tf.io.read_file(str(path))
    # Decode image (handles jpg, png, etc.) and assume 3 channels (RGB)
    # tf.image.decode_image automatically detects format but requires explicit channels for consistency
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    # Convert to float32 and scale to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Note: PyTorch code permutes to (C, H, W). TensorFlow uses (H, W, C) by default.
    # We keep (H, W, C) for standard TensorFlow model compatibility.
    return image

# Conversion: Replaced cv2 and torch operations with tf.io and tf.image
def read_mask(path: str | Path) -> tf.Tensor:
    # Read file
    mask = tf.io.read_file(str(path))
    # Decode as grayscale (1 channel)
    mask = tf.image.decode_png(mask, channels=1)
    # Cast to int8 to match original behavior
    mask = tf.cast(mask, tf.int8)
    return mask

class SegDataset:
    """Mask dataset for segmentation."""
    
    def __init__(
        self, 
        image_dir: str | Path,
        mask_dir: str | Path, 
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = list(self.image_dir.glob("*"))
        self.image_paths = [str(i) for i in image_paths if is_image_file(i)]

        self.mask_paths = []
        for img_p in self.image_paths:
            file_name = os.path.splitext(os.path.basename(img_p))[0]
            self.mask_paths.append(os.path.join(str(self.mask_dir), file_name + ".png"))
    
    # Conversion: __getitem__ is replaced by a mapping function for tf.data
    def _load_data(self, image_path, mask_path):
        image = read_image(image_path)
        mask = read_mask(mask_path)
        # PyTorch: mask.long() -> TensorFlow: tf.int64
        return {"image": image, "mask": tf.cast(mask, tf.int64)}
    
    def get_dataset(self):
        # Create a dataset from slices of file paths
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        # Map the loading function
        dataset = dataset.map(self._load_data, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(self.image_paths)


class InfiniteSampler:
    # Conversion: In PyTorch, this yields indices. In TensorFlow, we apply 
    # shuffling and repeating directly to the dataset pipeline.
    
    def __init__(self, data_source: SegDataset, shuffle: bool = True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.num_samples = len(data_source)

    def apply(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self.shuffle:
            # Shuffle the dataset. buffer_size=num_samples ensures full shuffling.
            dataset = dataset.shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)
        # Repeat indefinitely to mimic infinite sampling
        return dataset.repeat()

    def __len__(self):
        return self.num_samples

class DataLoader:
    # Conversion: Wrapper for tf.data.Dataset batching and prefetching
    def __init__(
        self, 
        dataset: SegDataset,
        batch_size: int,
        sampler: InfiniteSampler = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        # pin_memory and persistent_workers are handled automatically by tf.data runtime

    def __iter__(self):
        ds = self.dataset.get_dataset()
        
        if self.sampler:
            ds = self.sampler.apply(ds)
        
        # Batch the data
        ds = ds.batch(self.batch_size)
        
        # Prefetching overlaps data preprocessing and model execution
        # num_workers maps to AUTOTUNE for optimal parallelism
        if self.num_workers > 0:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return iter(ds)

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    sampler = InfiniteSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Conversion: itertools.cycle is not needed because InfiniteSampler applies .repeat()
    # We create an iterator from the DataLoader
    dataloader_iter = iter(dataloader)
    
    for step in range(300):
        batch = next(dataloader_iter)
        # batch is a dict {'image': Tensor(B, H, W, C), 'mask': Tensor(B, H, W, 1)}
```