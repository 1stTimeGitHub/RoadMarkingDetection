import torch
import pytorch_lightning as pl

from torchmetrics import F1Score

class SegmentationNetwork(pl.LightningModule):
    def __init__(self, net, lr=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.lr = lr
        self.f1score = F1Score(num_classes=20, average="macro", mdmc_average="global")

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_idx):
        _, loss, score = self._get_preds_loss_score(batch)
        self.log('Train Score', score)
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss, score = self._get_preds_loss_score(batch)
        self.log('Validation Score', score)
        self.log('Validation Loss', loss)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def _get_preds_loss_score(self, batch):
        img = batch['image']
        mask = batch['mask']
        pred = self.net(img)

        score = self.f1score(pred, mask)
        loss = torch.ones(1, requires_grad=True).to(self.device) - score

        return pred, loss, score

