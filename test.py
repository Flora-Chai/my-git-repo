from flcore.client.client import Client
import torch
import numpy as np
from os import path
from utils.MIA_attack import MIA_data
from HyperPrams import get_parser
import yaml
from utils.get_path import *
from utils.data.data_augment import data_aug
from torch.utils.data import DataLoader
args = get_parser()
config = yaml.load(
    open(get_config_path()),
    Loader=yaml.FullLoader,
)
client = Client(0)
'''
shadow_model.pt 不存在，文件夹的name拼写不正确
client.model.load_state_dict(torch.load("result/MNISTNOIID/third_rest_30_10_64_16_16_200_0_0.1__0.05_0.5/shadow_model.pt"))
data_in = np.load(path.join("npz_data/MNISTNOIID/0", "data_batch_1.npz"))["x"]
data_out = np.load(path.join("npz_data/MNISTNOIID/0", "data_batch_2.npz"))["x"]
'''
client.model.load_state_dict(torch.load("result/MNISTNONIID/third_rest_30_10_64_16_16_200_0_0.1__0.05_0.5/shadow_model.pt"))
data_in = np.load(path.join("npz_data/MNISTNONIID/0", "data_batch_1.npz"))["x"]
data_out = np.load(path.join("npz_data/MNISTNONIID/0", "data_batch_2.npz"))["x"]
in_data = torch.tensor(data_in, dtype=torch.float32)
out_data = torch.tensor(data_out, dtype=torch.float32)
label_in = np.load(path.join("npz_data/MNISTNONIID/0", "data_batch_1.npz"))["y"]
label_out = np.load(path.join("npz_data/MNISTNONIID/0", "data_batch_2.npz"))["y"]
in_label = torch.tensor(data_in, dtype=torch.float32)
out_label = torch.tensor(data_out, dtype=torch.float32)
InData = data_aug((in_data, in_label), args, "test")
OutData = data_aug((out_data, out_label), args, "test")

Inloader = DataLoader(InData, batch_size=config["mia_batch_size"])
Outloader = DataLoader(OutData, batch_size=config["mia_batch_size"])

