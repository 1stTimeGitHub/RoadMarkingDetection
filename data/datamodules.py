import pytorch_lightning as pl

from typing import Optional
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import RoadSegmentationDataset

class RoadImageDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=4, scale=1):
        self.batch_size = batch_size
        self.scale = scale

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transform = transforms.Compose([
                transforms.Resize([int(720*self.scale), int(1280*self.scale)])
            ])

            self.train = RoadSegmentationDataset('bdd100k/images/10k/train', 'bdd100k/labels/sem_seg/colormaps/train', train=True, scale=self.scale ,transform=train_transform)
            self.val = RoadSegmentationDataset('bdd100k/images/10k/val', 'bdd100k/labels/sem_seg/colormaps/val', train=False, scale=self.scale)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
