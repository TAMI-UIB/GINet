import hydra

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary
import rootutils

from src.model.GINet import GINet

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):

    data_loader = instantiate(cfg.dataset.datamodule)
    experiment = GINet(cfg)

    trainer = Trainer(max_epochs=cfg.model.train.max_epochs,
                      devices=cfg.devices,
                      callbacks=[ RichModelSummary(max_depth=4)],
                      )

    trainer.fit(experiment, train_dataloaders=data_loader)


if __name__ == '__main__':
    train()
