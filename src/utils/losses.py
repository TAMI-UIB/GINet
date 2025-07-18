import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.fft import fft2 as fft2
from torch.fft import ifft2 as ifft2

EPSILON = 1e-8

class L1(torch.nn.Module):
    def __init__(self, ):
        super(L1, self).__init__()
        self.L1 = torch.nn.L1Loss()

    def forward(self, output, input):
        pred = output['pred']
        gt = input['gt']
        l1 = self.L1(pred, gt)
        return l1