import glob
import os

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import rasterio
import random
import torchvision.transforms.functional as TF
from PIL import Image

from src.utils.patchwork import CreatePatches


class SentinelDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # print(data.keys())
        print(data['gt'].size())

    def __len__(self):
        return self.data['pan'].size(0)

    def __getitem__(self, idx):
        gt = self.data['gt'][idx]
        hs = self.data['hs'][idx]
        pan = self.data['pan'][idx]
        ms = self.data['ms'][idx]

        return dict(gt=gt, hs=hs, pan=pan, ms=ms)

    @staticmethod
    def get_rgb(hs):
        return hs[:, [0, 1, 2], :, :]


class Sentinel2(pl.LightningDataModule):
    def __init__(self, dataset_path='./data', test_path='test', batch_size=32, num_workers=4, patch_size=120, sampling=6):
        super().__init__()
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, 'train')
        self.val_path = os.path.join(dataset_path, 'val')
        self.test_path = os.path.join(dataset_path, test_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.sampling = sampling

        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.set_names = None


    def load_split(self, split_path):
        img_names = [name for name in os.listdir(split_path)]
        R10imgs, R20imgs = [], []

        R10_bands = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2']
        R20_bands = ['B05_20m.jp2', 'B06_20m.jp2', 'B07_20m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']

        for name in img_names:
            r10m_folder = os.path.join(split_path, name, 'R10m')
            r20m_folder = os.path.join(split_path, name, 'R20m')

            R10files = os.listdir(r10m_folder)
            R20files = os.listdir(r20m_folder)

            r10bands = self.load_bands(R10files, R10_bands, r10m_folder)
            R20bands = self.load_bands(R20files, R20_bands, r20m_folder)

            R10imgs.append(torch.stack([torch.from_numpy(band).unsqueeze(0) for band in r10bands], dim=1))
            R20imgs.append(torch.stack([torch.from_numpy(band).unsqueeze(0) for band in R20bands], dim=1))

        gt_list, hs_list, ms_list, pan_list = [], [], [], []
        #print(img_names)
        for i in range(len(R10imgs)):
            #print(i)
            _, _, h, w = R10imgs[i].shape
            new_h = h - (h % self.sampling)
            new_w = w - (w % self.sampling)
            R10imgs[i] = R10imgs[i][:, :, :new_h, :new_w]
            ms = self._downsampling(R10imgs[i], self.sampling)
            hs = R20imgs[i]
            gt = hs

            gt,ms = self._adapt_size(gt,ms)

            gt_patcher = CreatePatches(gt, self.patch_size, overlapping=60)
            ms_patcher = CreatePatches(ms, self.patch_size, overlapping=60)

            gt = gt_patcher.do_patches(gt)
            ms = ms_patcher.do_patches(ms)

            pan = self.classical_pan(ms)


            hs = self._downsampling(gt, self.sampling)

            if hs.size(0) != gt.size(0) or ms.size(0) != gt.size(0):
                return ValueError("The number of patches must be the same for all the images")

            hs_list.append(hs)
            ms_list.append(ms)
            pan_list.append(pan)
            gt_list.append(gt)

        hs = torch.cat(hs_list, dim=0).to(torch.float32)
        ms = torch.cat(ms_list, dim=0).to(torch.float32)
        pan = torch.cat(pan_list, dim=0).to(torch.float32)
        gt = torch.cat(gt_list, dim=0).to(torch.float32)

        return dict(gt=gt, hs=hs, pan=pan, ms=ms)

    def setup(self, stage="test"):
        train_data = self.load_split(self.train_path)
        val_data = self.load_split(self.val_path)
        test_data = self.load_split(self.test_path)

        self.train_dataset = SentinelDataset(train_data)
        self.val_dataset = SentinelDataset(val_data)
        self.test_dataset = SentinelDataset(test_data)


    @staticmethod
    def read_band(file_path):
        with rasterio.open(file_path) as src:
            band = src.read(1).astype(float)/ 10000.0
        return band

    def load_bands(self, band_filenames, band_names, folder_path):
        bands = []
        for band_name in band_names:
            band_file = next((file for file in band_filenames if band_name in file), None)
            if band_file:
                bands.append(self.read_band(os.path.join(folder_path, band_file)))
        return bands

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,   num_workers=self.num_workers)

    def val_dataloader(self):
        #return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return [
            DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers),
            DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        ]
    def classical_pan(self, ms):
        pan = torch.zeros_like(ms)
        pan = torch.cat([pan, pan[:, :2, :, :]], dim=1)

        canal8 = ms[:, 3, :, :].unsqueeze(1)

        canal48 = ms[:, [2,3], :, :]

        canal48mean = canal48.mean(dim=1, keepdim=True)

        canal8repeat = canal8.repeat(1, 3, 1, 1)
        canal48meanrepeat = canal48mean.repeat(1, 3, 1, 1)


        pan[:,3:6,:,:] = canal8repeat
        pan[:, 0:3, :, :] = canal48meanrepeat

        return pan

    def _adapt_size(self, hs, pan):
        N, C, H , W = hs.size()
        k = H // self.patch_size
        H_l = (H - k*self.patch_size) // 2
        H_r = (H-k*self.patch_size) - H_l
        W_l = (W - k*self.patch_size) // 2
        W_r = (W-k*self.patch_size) - W_l
        hs = hs[:, :, H_l:(H-H_r), W_l:(W-W_r)]

        N, C, H, W = pan.size()
        k = H // self.patch_size
        H_l = (H - k * self.patch_size) // 2
        H_r = (H - k * self.patch_size) - H_l
        W_l = (W - k * self.patch_size) // 2
        W_r = (W - k * self.patch_size) - W_l
        pan = pan[:, :, H_l:(H-H_r), W_l:(W-W_r)]

        return hs, pan

    @staticmethod
    def _downsampling(data, factor):
        low = torch.zeros(data.size(0), data.size(1), data.size(2) // factor, data.size(3) // factor)
        for i in range(factor):
            for j in range(factor):
                low += data[:, :, i::factor, j::factor] / factor ** 2
        return low
