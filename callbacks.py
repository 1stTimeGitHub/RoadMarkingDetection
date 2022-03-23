import wandb
import torch
import numpy as np

from PIL import Image
from pytorch_lightning.callbacks import Callback
 
class LogPredictionsCallback(Callback):

    def __init__(self):
        self.palette = np.load('palette.npy').tolist()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Once a batch is validated this is executed.
        """
        #Logs a table containing the image, ground truth, and prediction. Only the batch with idx 0 is logged
        if batch_idx == 0:
            #Gets the logger from the trainer
            logger = trainer.logger
            imgs = batch['image']
            masks = batch['mask']

            masks_nd = masks.squeeze(1).cpu().numpy().astype(np.uint8)
            outputs_nd = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)

            columns = ['image', 'ground truth', 'prediction']
            data = []

            for i in range(len(imgs)):
                img_nd = imgs[i].cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                img_pil = Image.fromarray(img_nd, mode='RGB')

                mask = Image.fromarray(masks_nd[i], mode='P')
                output = Image.fromarray(outputs_nd[i], mode='P')
                #Adds the color palette to the masks
                mask.putpalette(self.palette)
                output.putpalette(self.palette)

                data.append([wandb.Image(img_pil), wandb.Image(mask), wandb.Image(output)])

            #Logs table
            logger.log_table(key='table', columns=columns, data=data)