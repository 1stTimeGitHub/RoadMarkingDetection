import os
from xmlrpc.client import Boolean
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from PIL import Image

class RoadSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train: Boolean, scale: float = 1, transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train
        self.scale = scale
        self.transform = transform
        self.ids = [file.split('.')[0] for file in os.listdir(self.img_dir)]
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]

        img = self.openimage(self.img_dir, id)
        mask = self.openimage(self.mask_dir, id)

        assert img.size == mask.size, 'Image and mask {id} should be the same size, but are {img.size} and {cm.size}'

        img, mask = self.preprocess(img, mask)

        return {
            'image': img,
            'mask': mask
        }

    def preprocess(self, img, mask):
        imgnd = np.array(img)
        imgtensor = torch.from_numpy(imgnd.astype(np.float32).transpose((2,0,1)))
        masknd = np.array(mask).astype(np.int64)
        masknd[masknd == 255] = 19
        masktensor = torch.from_numpy(masknd)
        masktensor = masktensor.unsqueeze(0)

        if self.transform:
            imgtensor = self.transform(imgtensor)
            #Shape (1,H,W)
            masktensor = self.transform(masktensor)        

        return imgtensor, masktensor 


    def openimage(self, dir, id):
        img_file = glob(dir + '/' + id + '.*')
        assert len(img_file) == 1, \
            f'Either no image/colormap or multiple images/colormaps found for the ID {id}: {img_file}'
        return Image.open(img_file[0])