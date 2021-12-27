import numpy as np
import torch
import torch.utils.data as td
import pytorch_lightning as pl
import random
import os
import warnings

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class SantaDataset(td.Dataset):

    def __init__(self, root_dir, **kwargs):
        
        # Will store the data
        self.samples = list()
        
        # Get the transformation to be applied
        self.transform = kwargs.get('transform', None)

        # Get the kind of the data
        kind = kwargs.get('kind', None)
        assert kind in ['train', 'test'], "Sorry expected kind to be either `train` or `test`. Found some other value."

        desc = kwargs.get('desc', '')

        # The binary classes
        classes = ['not-a-santa', 'santa']

        for i in classes:
            
            path = os.path.join(root_dir, f'is_that_santa_data/{kind}/{i}')

            for img in tqdm(os.listdir(path), total=len(os.listdir(path)), desc=desc):
                img = os.path.join(path, img)
                x = Image.open(img)
                x = x.convert('RGB')
                y = classes.index(i)

                if self.transform is not None:
                    x = self.transform(x)

                self.samples.append((x, y))

        if kind == "train":
            random.shuffle(self.samples)
        

    def __len__(self):
        # Returns the total size of the dataset
        return len(self.samples)


    def __getitem__(self, index):
        # Returns a sample of the dataset for the specific index
        return self.samples[index]





class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        
        super().__init__()

        # Get the data path
        self.path = kwargs.get('path', None)
        assert self.path is not None, "Sorry cannot proceed as the path is not provided"

        # Get the batch size
        self.batch_sz = kwargs.get('batch_size', 64)

        # Get the training size
        self.train_percent = kwargs.get('train_percent', 0.8)

        # Define the transformations 
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Grayscale(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        ])


        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.0), (1.0))
        ])


    def prepare_data(self):
        
        # Check if data/ is present 
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Check for downloaded files
        if not len(os.listdir(self.path)):
            # Download the data from Kaggle
            os.system(f"kaggle datasets download -d deepcontractor/is-that-santa-image-classification --path {self.path}")

            # Unzip the downloaded data
            os.system(f"unzip {self.path}/*.zip -d {self.path}/ && mv {self.path}/is\ that\ santa/ {self.path}/is_that_santa_data/")

            # Remove the zip files
            os.system(f"rm -rf {self.path}/*.zip")

    
    def setup(self, stage=None):
        

        if stage == 'fit' or stage is None:
            # Get the training data and validation data
            train_dataset = SantaDataset(self.path, transform=self.train_transform, kind="train", desc="Processing Training Data : ")

            train_size = int(self.train_percent * train_dataset.__len__())
            val_size = train_dataset.__len__() - train_size

            print(f'Spliting data into [Train-set : {train_size} samples, Validation-set : {val_size} samples]')
            self.train_dataset, self.val_dataset = td.random_split(train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:

            # Get the testing data
            self.test_dataset = SantaDataset(self.path, transform=self.test_transform, kind="test", desc="Processing Test Data : ")

    def train_dataloader(self):
        return td.DataLoader(self.train_dataset, batch_size=self.batch_sz, shuffle=True)

    def val_dataloader(self):
        return td.DataLoader(self.val_dataset, batch_size=self.batch_sz, shuffle=False)

    def test_dataloader(self):
        return td.DataLoader(self.test_dataset, batch_size=1, shuffle=False)



