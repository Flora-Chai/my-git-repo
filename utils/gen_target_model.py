
import torch
import numpy as np
import pandas as pd
import os
from HyperPrams import get_parser
from os import path
from utils.get_path import *
from flcore.client.client import Client


def generate_target_model(server, timer, mode):
    # 读取配置信息
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    client_num = config["clients"]
    clients = []
    args = get_parser()

    # 初始化客户端列表
    for i in range(client_num):
        clients.append(Client(i))

    # 将客户端模型加载为服务器模型的状态字典
    for i in range(len(clients)):
        clients[i].model.load_state_dict(
            server.model.state_dict()
        )

    # 重新获取命令行参数
    args = get_parser()
    acc_D_fs = []
    acc_D_rs = []
    acc_D_tests = []
    train_times = []

    # 遍历每个客户端进行测试
    for i in range(len(clients)):
        # 测试遗忘数据部分的准确率
        acc_D_f = clients[i].test("forgotten")
        # 测试剩余数据部分的准确率
        acc_D_r = clients[i].test("rest")
        # 测试测试数据部分的准确率
        acc_D_test = clients[i].test("test")
        acc_D_fs.append(acc_D_f)
        acc_D_rs.append(acc_D_r)
        acc_D_tests.append(acc_D_test)
        train_times.append(timer)

    # 计算平均准确率并添加到列表
    acc_D_fs.append(np.mean(np.array(acc_D_fs)))
    acc_D_rs.append(np.mean(np.array(acc_D_rs)))
    acc_D_tests.append(np.mean(np.array(acc_D_tests)))
    train_times.append(np.mean(np.array(train_times)))

    # 创建包含测试结果的DataFrame
    df = pd.DataFrame(
        {
            "D_f": acc_D_fs,
            "D_r": acc_D_rs,
            "D_test": acc_D_tests,
            "train_time": train_times,
        }
    )

    # 检查模型保存路径是否存在，如果不存在则创建
    if not path.exists(model_save_path()):
        os.makedirs(model_save_path())

    # 保存服务器模型的状态字典和测试结果到文件
    torch.save(
        server.model.state_dict(),
        path.join(model_save_path(), f"{mode}_model.pt"),
    )
    df.to_csv(
        path.join(model_save_path(), f"{mode}_train_results.csv"),
        index=True,
        float_format="%.4f",
    )

