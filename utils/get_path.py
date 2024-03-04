from HyperPrams import get_parser
import yaml
import os
from os import path

# 设置根路径
# ROOT_PATH = "/home/zhangdongping/project/normal_fed/fu_subset"
ROOT_PATH = r"C:\Users\DELL\Desktop\documents\FU HAR\fu_subset"

# 获取模型保存路径
def get_model_path():
    return path.join(ROOT_PATH, "result")

# 获取配置文件路径
def get_config_path():
    dataset = get_parser().dataset
    return path.join(ROOT_PATH, "config", f"{dataset}.yaml")

# 获取数据集路径
def get_data_path():
    dataset = get_parser().dataset
    return path.join(ROOT_PATH, "npz_data", f"{dataset}")

# 获取模型保存路径，ROOT_PATH, "result"
def model_save_path():
    args = get_parser()
    dataset = args.dataset
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    return path.join(
        get_model_path(),
        dataset,
        f"{config['forget_data']}_{config['balance_data']}_"
        + f"{config['server_epochs']}_{config['client_epochs']}_"
        + f"{config['train_batch_size']}_{config['forget_batch_size']}_"
        + f"{config['balance_batch_size']}_{args.forget_size}_{args.id}_"
        + f"{config['rate']}_{config['ur']}_{config['frac']}_{config['note']}",
    )
#\third_rest_10_5_128_16_16_200_0_0.1_0.01_0.5_xx\0\unlearn_0.01_model.pt

# 获取图形保存路径
def graph_save_path():
    args = get_parser()
    dataset = args.dataset
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    return path.join(
        get_model_path(),
        dataset,
        f"{config['forget_data']}_{config['balance_data']}_"
        + f"{config['server_epochs']}_{config['client_epochs']}_"
        + f"{config['train_batch_size']}_{config['forget_batch_size']}_"
        + f"{config['balance_batch_size']}_{args.forget_size}_{args.id}_"
        + f"{config['rate']}_{config['ur']}_{config['frac']}_{config['note']}",
    )
