from pathlib import Path
from typing import Union
import gdown
import zipfile
import os

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

class CatDogImageDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        # Google Drive file ID (from the shareable link)

        # URL to download the file
        download_url = f"https://drive.google.com/uc?id=1X96sA534xC5_Yh37jhYSKVQ0PFtJLNNn"

        # Name of the output zip file (it will be downloaded in the current directory)
        output_zip = 'cat_images.zip'

        # Download the zip file
        gdown.download(download_url, output_zip, quiet=False)

        # Extract the zip file in the current directory
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
           zip_ref.extractall(".")  # Extract to the current directory

        # Optionally, remove the downloaded zip file after extraction
        os.remove(output_zip)

        print("Cat images downloaded and extracted successfully.")

    @property
    def data_path(self):
        return Path(self._dl_path)

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation/test loaders."""
        if train:
            dataset = self.create_dataset(self.data_path, self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path, self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)  # Using validation dataset for testing