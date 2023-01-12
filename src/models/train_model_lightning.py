import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
import sys
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback #importing Callbacks class

from src.data.data import CorruptMnistDataModule
from srcmodels.model import MyAwesomeModel

import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")
   
    wandb.init(project="final_ex", config= config)
    
    data_module = CorruptMnistDataModule(
        os.path.join(hydra.utils.get_original_cwd(), config.data.path),###
        batch_size=config.train.batch_size,
    )
    data_module.setup()
    model = MyAwesomeModel(config)

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=config.train.epochs,
        logger=pl.loggers.WandbLogger(project="final_ex", config=config),#Add the wandb logger
        enable_checkpointing=True, #Callbacks ModelCheckpoint
        callbacks=[checkpoint_callback],
        limit_test_batches=0.25, # run through only 25% of the test set each epoch
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        #val_dataloaders=data_module.test_dataloader(),
    )

    torch.save(model.state_dict(), 'src/models/trained_model_laightning.pt')           


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()