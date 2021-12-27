import numpy as np
import torch
import torch.utils.data as td
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import yaml
import sys
import warnings

from PIL import Image
from torchvision import transforms
from scripts import model


warnings.filterwarnings('ignore')

if __name__ == "__main__":

    try:
        with open('configs/config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

    except yaml.YAMLError as exc:
        print(exc)

    img = Image.open('./sample_test_imgs/not-a-santa.jpeg').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.))

    ])

    img = transform(img)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    img = img.unsqueeze(0)



    # Create a dataset
    dataset = td.TensorDataset(img)
    dataloader = td.DataLoader(dataset)

    classifier = model.LitClassifier.load_from_checkpoint('models/epoch=16-step=84.ckpt', cfg=cfg)
    classifier.freeze()
    trainer = pl.Trainer(accelerator='cpu')
    pred = trainer.test(classifier, dataloader)[0]
    print('It is a Santa' if pred['prediction'] else 'Not a Santa')
    


