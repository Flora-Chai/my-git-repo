from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import torch


def get_transform(dataset, mode):
    if dataset in ["MNIST", "FASHION", "MNISTNONIID"]:
        if mode == "train":
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),
                ]
            )

    return transform


class data_aug(Dataset):
    def __init__(self, data, args, mode):
        self.mode = mode
        self.args = args
        self.x = data[0]
        self.y = torch.tensor(data[1], dtype=torch.int64)
        self.transform = get_transform(self.args.dataset, self.mode)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        img = self.transform(self.x[idx])
        label = self.y[idx]

        return img, label
        # return img
