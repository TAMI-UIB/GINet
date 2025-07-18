from typing import Dict, Any

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
import torch
import pytorch_lightning as pl

from src.model.modules.operators import ResNL, ClustersUp, Down, Upsampling


class GINet(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define subsets
        self.loss_criterion = instantiate(cfg.model.train.loss)

        self.sampling = cfg.sampling
        self.classes = cfg.model.module.classes
        self.BPKernel = nn.ModuleList([
            ResNL(u_channels=cfg.model.module.hs_channels, pan_channels=cfg.model.module.hs_channels, features_channels=cfg.model.module.features,
                                          patch_size=cfg.model.module.patch_size, window_size=cfg.model.module.window_size, kernel_size=cfg.model.module.kernel_size)
            for i in range(cfg.model.module.iter_stages)
        ])
        self.iter_stages = cfg.model.module.iter_stages
        self.plus = cfg.model.module.plus
        self.spectral_inter = ClustersUp(cfg.model.module.ms_channels, cfg.model.module.hs_channels, cfg.model.module.classes, cfg.model.module.features)
        self.clustering_cnn = nn.Sequential(
            nn.Conv2d(cfg.model.module.ms_channels, 32, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.Conv2d(64, self.classes, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.Softmax(dim=1)  # Probabilidades de pertenencia a cada clúster
        )
        self.downsamp_hs = Down(channels=cfg.model.module.hs_channels, sampling=cfg.sampling)
        self.upsamp_hs = Upsampling(cfg.model.module.hs_channels, cfg.sampling)
        self.downsamp_hs = nn.Sequential(
            *[Down(channels=cfg.model.module.hs_channels, sampling=cfg.sampling) for _ in range(self.iter_stages)])
        self.upsamp_hs = nn.Sequential(*[Upsampling(cfg.model.module.hs_channels, cfg.sampling) for _ in range(self.iter_stages)])


    def forward(self, input):
        ms = input["ms"]
        hs = input["hs"]
        if self.plus:
            clusters_probs = self.clustering_cnn(ms)
            clusters = torch.argmax(clusters_probs, dim=1, keepdim=True)
            pan = self.spectral_inter(ms, clusters)
        else:
            pan = input["pan"]
        u = nn.functional.interpolate(hs, scale_factor=self.sampling, mode='bicubic')
        for i in range(self.iter_stages):
            DBu = self.downsamp_hs[i](u)
            error = self.upsamp_hs[i](DBu - hs)
            u = u + self.BPKernel[i](error, pan)
        return {"pred": u}

    def training_step(self, input, idx):
        output = self.forward(input=input)
        loss = self.loss_criterion(output, input)
        return {"loss": loss, "output": output}

    def validation_step(self, input, idx, dataloader_idx=0):
        output = self.forward(input=input)
        loss = self.loss_criterion(output, input)
        return {"loss": loss, "output": output}

    def test_step(self, input, idx,  dataloader_idx=0):
        output = self.forward(input=input)
        return output

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.train.optimizer,params=self.parameters())
        scheduler = instantiate(self.cfg.model.train.scheduler, optimizer=optimizer)
        return {'optimizer': optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['cfg'] = self.cfg
        checkpoint['current_epoch'] = self.current_epoch
