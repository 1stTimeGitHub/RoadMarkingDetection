import argparse
import logging
import os
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image

import segmentation_models_pytorch as smp

def segment_road(net, device, imgdir, out):
    palette = np.load('palette.npy').tolist()

    for fname in os.listdir(imgdir):
        img = Image.open(fname)
        imgnd = np.array(img)
        imgnd = imgnd.astype(np.float32)
        imgnd = imgnd.transpose((2,0,1))
        imgtensor = torch.from_numpy(imgnd)

        imgtensor = imgtensor.to(device, dtype=torch.float32)

        with torch.no_grad():
            pred = net(imgtensor)

        prednd = torch.argmax(pred, dim=1).cpu().numpy().astype(np.uint8)
        predimg = Image.fromarray(prednd, mode='P')
        predimg.putpalette(palette)
        predimg.save(out + '/' + fname)
        

    print("Segmenting Road")

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--checkpoint', help='Loads a checkpoint', dest='checkpoint', type=str, required=True)
    parser.add_argument('-enc', '--encoder', help='Encoder to be used', dest='encoder', type=str, required=True)
    parser.add_argument('-net', '--network', help='Decoder to be used', dest='network', type=str, required=True)
    parser.add_argument('-i', '--imgs-directory', help='Directory where the images are stored', dest='imgs', type=str, required=True)
    parser.add_argument('-o', '--output-directory', help='Directory where the output will be stored', dest='out', type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()

    if args.network == 'pspnet':
        net = smp.PSPNet(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=20)
    elif args.network == 'deeplabv3':
        net = smp.DeepLabV3(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=20)
    elif args.network == 'deeplabv3+':
        net = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=20)
    else:
        logging.warning('Desired network not supported!')
        exit()

    #Load checkpoint
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    segment_road(net, device, args.imgs, args.out)