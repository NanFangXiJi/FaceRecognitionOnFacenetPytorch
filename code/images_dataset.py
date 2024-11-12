import os
from torch.utils.data import Dataset

from process import load_image


class ImageDataset(Dataset):
    def __init__(self, image_folder, device, rotation=False):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if
                            fname.endswith(('jpg', 'jpeg', 'png'))]
        self.device = device
        self.rotation = rotation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = load_image(img_path, self.rotation)
        filename = os.path.basename(img_path)
        return image, filename
