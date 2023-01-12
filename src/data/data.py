
import torch
#import wget
from torch.utils.data import Dataset
import numpy as np
import os


class CorruptMnist(Dataset):
    def __init__(self, train):
        path= "/home/denisa/MLOPS/final_ex/data/processed"
        # path= "/data/processed" -this should be enabled when running docker
        if train:
            data = torch.load(os.path.join(path, "data_train.pkl"))
            targets = torch.load(os.path.join(path, "targets_train.pkl"))
        else:
            data = torch.load(os.path.join(path, "data_test.pkl"))
            targets = torch.load(os.path.join(path, "targets_test.pkl"))

        self.data = data
        self.targets = targets


    def __len__(self):
        return self.targets.numel()

    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]

#Implementing Pytorch Lightning
class CorruptMnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = os.path.join(data_path, "processed")
        self.batch_size = batch_size
        self.cpu_cnt = os.cpu_count() or 2

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_path):
            raise Exception("data is not prepared")

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = CorruptMnist(self.data_path, "train")
        self.testset = CorruptMnist(self.data_path, "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.cpu_cnt
        )

if __name__ == "__main__":
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)
