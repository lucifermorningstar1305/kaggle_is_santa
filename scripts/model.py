import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics import Accuracy
from torchsummary import summary

from scripts.network_builder import build_network


class LitClassifier(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        # Get the total number of classes
        self.n_classes = cfg['n_classes']

        # Get the learning rate for the optimizer
        self.lr = cfg['learning_rate']

        # Build the model
        self.model = build_network(cfg)
        self.model.append(nn.LazyLinear(self.n_classes))
        self.model = nn.Sequential(*self.model)

        # Print a summary of the model
        summary(self.model.cuda(), (3, 256, 256))

        
        # Define the accuracy model
        self.accuracy = Accuracy(num_classes=self.n_classes)


    def forward(self, x):
        x = self.model(x)
        return F.sigmoid(x)


    def training_step(self, batch, batch_idx):

        # Get the data and it's corresponding label
        x, y = batch
        
        # Calculate the logit
        logits = self.forward(x)

        # Change the dimension of the label from (dim, ) to (dim, 1)
        y = y.view(y.size(0), 1)

        # Calculate the loss
        loss = F.binary_cross_entropy(logits, y.float())

        # Log the values
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        # Get the data and it's corresponding label
        x, y = batch

        # Calculate the logit
        logits = self.forward(x)

        # Change the dimension of the label from (dim, ) to (dim, 1)
        y = y.view(y.size(0), 1)

        # Calculate the loss
        loss = F.binary_cross_entropy(logits, y.float())

        # Calculate the accuracy
        self.accuracy(logits, y.int())

        # Log the values
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)


        # Log Images and it's caption
        imgs = x.to(device='cpu')
        labels = y.to(device='cpu').numpy()
        preds = logits.to(device='cpu').numpy()
        self.logger.log_image(
            key="validation", 
            images = [img for img in imgs], 
            caption=[f'Pred : {int(pred[0] > 0.5)}, Label : {label[0]}' for pred, label in zip(preds, labels)]
            )

        return loss


    def test_step(self, batch, batch_idx):
        
        # Get the data and it's corresponding label
        if len(batch) > 1:
            x, y = batch
        else:
            x, y = batch, None
            x = x[0]
            print(x.size())
            

        if y is not None:
            # Calculate the logit
            logits = self.forward(x)

            # Change the dimension of the label from (dim, ) to (dim, 1)
            y = y.view(y.size(0), 1)

            # Calculate the loss
            loss = F.binary_cross_entropy(logits, y.float())

            # Calculate the accuracy
            self.accuracy(logits, y.int())

            # Log the values
            self.log('test_loss', loss, prog_bar=True)
            self.log('test_acc', self.accuracy, prog_bar=True)

            return loss

        else:
            # Calculate the logit
            logits = self.forward(x)
            pred = int(logits.numpy().squeeze() > 0.5)
            self.log('prediction', pred)

            return pred



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1e-2, patience=15, verbose=True, min_lr=1e-8)

        return {
            "optimizer" : optimizer,
            "lr_scheduler" : {
                "scheduler" : lr_scheduler,
                "monitor" : "val_loss"
            }
        }



