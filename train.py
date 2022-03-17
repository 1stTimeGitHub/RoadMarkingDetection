import argparse
import logging
import wandb
import segmentation_models_pytorch as smp

from callbacks import LogPredictionsCallback
from network import SegmentationNetwork
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data.datamodules import RoadImageDataModule

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--train', metavar='TRN', type=str, default=None, help='Directory to the training set', dest='traindir')
    parser.add_argument('-v', '--validation', metavar='VAL', type=str, default=None, help='Directory to the validation set', dest='valdir')
    parser.add_argument('-net', '--network', metavar='NET', type=str, default=None, help='Network to be used', dest='network')
    parser.add_argument('-enc', '--encoder', metavar='ENC', type=str, default=None, help='Encoder to be used', dest='encoder')
    parser.add_argument('-s', '--scale', metavar='S', type=float, default=1, help='Image scale', dest='scale')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    
    wandb.login()
    wandb_logger = WandbLogger(project="RoadMarkingDetection", log_model="all")
    trainer = Trainer(gpus=1, logger=wandb_logger, callbacks=[LogPredictionsCallback()])

    if args.network == 'pspnet':
        net = smp.PSPNet(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=20)
    else:
        logging.warning('Desired network not supported')
        exit()

    net.encoder.eval()
    for m in net.encoder.modules():
        m.requires_grad_ = False

    model = SegmentationNetwork(net)
    wandb_logger.watch(model)

    dm = RoadImageDataModule(scale=args.scale)

    trainer.fit(model=model, datamodule=dm)