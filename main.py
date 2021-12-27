import numpy as np
import torch
import pytorch_lightning as pl
import argparse
import os
import sys
import yaml
import warnings

from scripts import datamodule, model

warnings.filterwarnings('ignore')


AVAIL_GPUS = min(1, torch.cuda.device_count())

pl.seed_everything(42)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', help='Defines the action (train/test) a model.')

    args = parser.parse_args()

    try:
        with open('./configs/config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

    except yaml.YAMLError as exc:
        print(exc)

    if args.action == 'train':
        # Load the DataModule
        dm = datamodule.DataModule(path=cfg['path'], batch_size=cfg['batch_size'], train_percent=0.98)

        # Create the model
        classifier = model.LitClassifier(cfg)

        # Declare the trainer
        trainer = pl.Trainer(
            gpus=AVAIL_GPUS, 
            max_epochs=cfg['epochs'], 
            callbacks = [
                pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=5, verbose=True),
                pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='val_loss', mode='min', dirpath='models/')])

        # Fit the model
        trainer.fit(classifier, dm)



    elif args.action == 'test':

        # Load the Data Module
        dm = datamodule.DataModule(path=cfg['path'], batch_size=1)
        dm.setup(stage='test')

        # Load the model from the checkpoint
        classifier = model.LitClassifier.load_from_checkpoint('models/epoch=16-step=84.ckpt', cfg=cfg)
        classifier.freeze()

        # Define the trainer
        trainer = pl.Trainer(accelerator='cpu')

        # Test the model on the test set
        trainer.test(classifier, dm)

    else:
        print(f'Sorry an action is required to proceed with the program.')