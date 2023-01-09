import glob
import logging
import os
import sys
from pathlib import Path
from time import time

import hydra
import numpy as np
import omegaconf
import torch.quantization
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from torch import nn

sys.path.append("/home/denisa/MLOPS/MNIST_repo/src")
from data.data import CorruptMnist
from models.model import MyAwesomeModel


#@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main():
    logger = logging.getLogger(__name__)
    logger.info("Executing predict model script.")

    output_dir = "/home/denisa/MLOPS/MNIST_repo/models"

    output_config_path= " /home/denisa/MLOPS/MNIST_repo/src/models"

    # Load model
    model= torch.load(os.path.join(output_dir, "trained_model.pt"))

    data_module = CorruptMnist(train=False)
    data_test= data_module.data

    output_prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(output_prediction_dir, exist_ok=True)

    pred= model(data_test.float())
    pred_np = pred.detach().numpy()    
    output_prediction_file = os.path.join(output_prediction_dir, "predictions.csv")
    np.savetxt(output_prediction_file, pred_np, delimiter=",")



if __name__ == "__main__":
    #log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()