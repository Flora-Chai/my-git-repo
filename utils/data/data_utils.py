# 导入必要的模块
from utils.data.data_loader import get_dataloader
import numpy as np
from HyperPrams import get_parser
from utils.get_path import *
from utils.data.data_augment import data_aug
from sklearn.model_selection import train_test_split


def read_client_data(idx, data_type="train"):
    # 获取命令行参数
    args = get_parser()
    # 获取数据路径
    data_path = get_data_path()

    if data_type == "train":
        # 读取训练数据
        x = np.load(f"{data_path}/{idx}/data_batch_0.npz")["x"]
        y = np.load(f"{data_path}/{idx}/data_batch_0.npz")["y"]
        # 如果模型类型为 "shadow"，则读取额外的数据
        # if args.model_type == "shadow":
        #     x = np.load(f"{data_path}/{idx}/data_batch_1.npz")["x"]
        #     y = np.load(f"{data_path}/{idx}/data_batch_1.npz")["y"]

        # 处理空标签的情况
        if y.shape[0] == 0:
            x = np.random.normal(0, 1, size=(1, x.shape[1], x.shape[2]))
            y = np.random.randint(0, 1, size=(1,))

        return data_aug((x, y), args, "train")
    elif data_type == "test":
        # 读取测试数据
        x = np.load(f"{data_path}/{idx}/test.npz")["x"]
        y = np.load(f"{data_path}/{idx}/test.npz")["y"]
        # 处理空标签的情况
        if y.shape[0] == 0:
            x = np.random.normal(0, 1, size=(1, x.shape[1], x.shape[2]))
            y = np.random.randint(0, 1, size=(1,))
        return data_aug((x, y), args, "test")
    elif data_type == "forgotten":
        # 读取遗忘数据
        x = np.load(f"{data_path}/{idx}/data_batch_0.npz")["x"][: args.forget_size]
        y = np.load(f"{data_path}/{idx}/data_batch_0.npz")["y"][: args.forget_size]
        # 处理空标签的情况
        if y.shape[0] == 0:
            x = np.random.normal(0, 1, size=(1, x.shape[1], x.shape[2]))
            y = np.random.randint(0, 1, size=(1,))
        return data_aug((x, y), args, "train")
    elif data_type == "rest":
        # 读取剩余数据
        x = np.load(f"{data_path}/{idx}/data_batch_0.npz")["x"][args.forget_size:]
        y = np.load(f"{data_path}/{idx}/data_batch_0.npz")["y"][args.forget_size:]
        # 处理空标签的情况
        if y.shape[0] == 0:
            x = np.random.normal(0, 1, size=(1, x.shape[1], x.shape[2]))
            y = np.random.randint(0, 1, size=(1,))
        return data_aug((x, y), args, "train")
    elif data_type == "third":
        # 读取第三方数据
        x = np.load(f"{data_path}/{idx}/data_batch_2.npz")["x"]
        y = np.load(f"{data_path}/{idx}/data_batch_2.npz")["y"]
        # 处理空标签的情况
        if y.shape[0] == 0:
            x = np.random.normal(0, 1, size=(1, x.shape[1], x.shape[2]))
            y = np.random.randint(0, 1, size=(1,))
        return data_aug((x, y), args, "train")
    else:
        print("dataset is not exist!")


def split(dataset_size, num_users=3):
    """
    Sample I.I.D. client data from FASHION dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    dict_users, all_idxs = {}, [i for i in range(dataset_size)]

    num_items = int(len(all_idxs) / num_users)

    for i in range(num_users):
        # 随机抽样创建用户数据集
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def data_split():
    # 获取命令行参数和配置信息
    args = get_parser()
    config = yaml.load(
        open(get_config_path()),
        Loader=yaml.FullLoader,
    )
    clients = config["clients"]
    data_path = get_data_path()

    for i in range(clients):
        # 读取训练数据和标签
        datasets = np.load(f"{data_path}/{i}/train.npz")["x"]
        targets = np.load(f"{data_path}/{i}/train.npz")["y"]

        # 将数据集分割为不同的用户数据集
        index = split(len(datasets), 3)

        for j in range(len(index)):
            indice = index[j]
            indice = np.array(list(index[j]))
            x = datasets[indice]
            y = targets[indice]

            # 根据标签排序数据
            sort_indices = np.argsort(y)
            x = x[sort_indices]
            y = y[sort_indices]

            # 保存分割后的数据集
            np.savez(
                f"{data_path}/{i}/data_batch_{j}.npz",
                x=x,
                y=y,
            )


