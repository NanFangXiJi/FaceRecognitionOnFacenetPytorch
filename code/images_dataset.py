import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, image_folder, k, transform=None):
        self.image_folder = image_folder
        self.k = k
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if
                            fname.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) // self.k

    def __getitem__(self, idx):
        start_idx = idx * self.k
        end_idx = start_idx + self.k

        images = []
        filenames = []

        for i in range(start_idx, end_idx):
            img_path = self.image_paths[i]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
            filenames.append(self.image_paths[i])

        return images, filenames
