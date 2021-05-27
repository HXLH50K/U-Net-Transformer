import os
import os.path as osp
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix

        self.ids = [
            osp.splitext(file)[0] for file in os.listdir(imgs_dir)
            if not file.startswith('.')
        ]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_nd):
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
            # img_nd = np.repeat(img_nd, 3, 2) # make 1 channel pic to 3 channels pic

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if 255 >= img_trans.max() > 1 and img_trans.min() > 0:
            # Normally UINT8 pic
            img_trans = img_trans / 255.0
        elif 0 < img_trans.all() <= 1:
            # Normally FLOAT pic
            pass
        else:
            # DICOM pic
            pass

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # print(
        #     self.masks_dir,
        #     idx,
        #     self.mask_suffix,  #0.5
        # )
        mask_file = glob(
            osp.join(self.masks_dir, idx + self.mask_suffix + '.*'))
        img_file = glob(osp.join(self.imgs_dir, idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0])
        img = np.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return torch.from_numpy(img).type(
            torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
