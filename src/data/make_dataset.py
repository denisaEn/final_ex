# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    train_data = []
    test_data = []

    for i in range(5):
        train_data.append(np.load(os.path.join(input_filepath, f"train_{i}.npz") , allow_pickle=True))
        data_train = torch.tensor(np.concatenate([c['images'] for c in train_data])).reshape(-1, 1, 28, 28)
        targets_train = torch.tensor(np.concatenate([c['labels'] for c in train_data]))

    test_data = np.load(os.path.join(input_filepath, "test.npz"), allow_pickle=True)
    data_test = torch.tensor(test_data['images']).reshape(-1, 1, 28, 28)
    data_test= data_test/data_train.max()
    targets_test = torch.tensor(test_data['labels'])
 
    torch.save(data_train, os.path.join(output_filepath, "data_train.pkl"))
    torch.save(data_test, os.path.join(output_filepath, "data_test.pkl"))
    torch.save(targets_train, os.path.join(output_filepath, "targets_train.pkl"))
    torch.save(targets_test, os.path.join(output_filepath, "targets_test.pkl"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
