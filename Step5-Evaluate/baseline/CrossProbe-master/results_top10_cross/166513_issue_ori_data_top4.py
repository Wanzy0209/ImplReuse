```python
import tensorflow as tf
import os
from pathlib import Path
import random
import itertools

# Conversion: Removed torch, cv2, numpy, gc imports as they are replaced by TensorFlow ops or standard Python libs

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_image_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS

# Conversion: Replaced cv2 and torch.from_numpy with tf.io and tf.image ops
def read_image(path: str | Path, as_tensor: bool = True) -> tf.Tensor:
    # Conversion: tf.io.read_file reads the file content
    file = tf.io.read_file(path)
    # Conversion: tf.image.decode_image decodes various formats (jpg, png, etc.)
    # expand_animations=False prevents creating a 4th dimension for gifs
    image = tf.image.decode_image(file, channels=3, expand_animations=False)
    # Conversion: convert_image_dtype scales uint8 [0, 255] to float32 [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Note: PyTorch code permutes to (C, H, W). TensorFlow uses (H, W, C) by default.
    # If the model specifically requires CHW, uncomment the following line:
    # image = tf.transpose(image, perm=[2, 0, 1])
    
    return image

# Conversion: Replaced cv2 and torch.from_numpy with tf.io and tf.image ops
def read_mask(path: str | Path, as_tensor: bool = True) -> tf.Tensor:
    file = tf.io.read_file(path)
    # Conversion: Decode as grayscale (1 channel)
    mask = tf.image.decode_png(file, channels=1)
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
            self.mask_paths.append(os.path.join(self.mask_dir, file_name + ".png"))
    
    # Conversion: __getitem__ is replaced by a mapping function in tf.data pipeline
    def decode(self, image_path: str, mask_path: str):
        image = read_image(image_path)
        mask = read_mask(mask_path)
        # Conversion: PyTorch .long() is int64
        return {"image": image, "mask": tf.cast(mask, tf.int64)}
    
    def __len__(self):
        return len(self.image_paths)

# Conversion: InfiniteSampler is replaced by .repeat() and .shuffle() in tf.data
# No class definition needed for the sampler in TensorFlow.

if __name__ == "__main__":
    image_dir = ""
    mask_dir = ""
    dataset = SegDataset(image_dir, mask_dir)
    
    # Conversion: Create a tf.data.Dataset from the file paths
    # This replaces the instantiation of the PyTorch Dataset
    file_dataset = tf.data.Dataset.from_tensor_slices((dataset.image_paths, dataset.mask_paths))
    
    # Conversion: InfiniteSampler logic
    # 1. Shuffle: mimics shuffling indices. reshuffle_each_iteration=True mimics reshuffling every epoch.
    # 2. Repeat: mimics the infinite nature of InfiniteSampler.
    file_dataset = file_dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
    file_dataset = file_dataset.repeat()
    
    # Conversion: Map the decode function (equivalent to __getitem__)
    # num_parallel_calls=tf.data.AUTOTUNE mimics num_workers
    tf_dataset = file_dataset.map(dataset.decode, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Conversion: DataLoader parameters
    # batch_size=12 -> .batch(12)
    # pin_memory=True, persistent_workers=True -> Handled automatically by tf.data runtime
    tf_dataset = tf_dataset.batch(12)
    
    # Conversion: .prefetch(tf.data.AUTOTUNE) optimizes performance, similar to pin_memory/prefetching
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Conversion: itertools.cycle is not needed because .repeat() makes the dataset infinite
    infinite_dataloader = tf_dataset
    
    for step, batch in enumerate(infinite_dataloader):
        if step >= 300:
            break
        # batch is a dict {'image': Tensor, 'mask': Tensor}
        pass
```