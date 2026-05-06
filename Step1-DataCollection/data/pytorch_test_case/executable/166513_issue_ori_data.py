import torch
from torch.utils.data import Sampler, DataLoader, Dataset
import itertools
import random
import gc
import cv2
import numpy as np
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def is_image_file(filename: str | Path) -> bool:
    filename = Path(filename)
    return filename.suffix.lower() in IMG_EXTENSIONS

def read_image(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    if as_tensor:
        return torch.from_numpy(image).permute(2, 0, 1)
    return image

def read_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.int8)
    if as_tensor:
        mask = torch.from_numpy(mask)
    return mask

class SegDataset(Dataset):
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


class InfiniteSampler(Sampler):
    
    def __init__(self, data_source: Dataset, shuffle: bool = True):
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
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    infinite_dataloader = itertools.cycle(dataloader)
    for step in range(300):
        batch = next(infinite_dataloader)