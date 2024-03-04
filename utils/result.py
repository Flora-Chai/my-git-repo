import torch
import time
import numpy as np
from os import path
import pandas as pd
from utils.get_path import *
from flcore.client.client import Client


def result_unlearn(args, lam):
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    client_num = config["clients"]
    clients = []
    for i in range(client_num):
        clients.append(Client(i))
    for i in range(len(clients)):
        clients[i].model.load_state_dict(
            torch.load(
                os.path.join(model_save_path(), f"{args.id}", f"unlearn_{lam}_model.pt")
            )
        )

    acc_D_fs = []
    acc_D_rs = []
    acc_D_tests = []
    train_times = []
    for i in range(len(clients)):
        # 要遗忘部分数据的测试准确率
        time1 = time.time()
        acc_D_f = clients[i].test("forgotten")
        # 剩余部分数据的测试准确率
        acc_D_r = clients[i].test("rest")
        # 测试数据的测试准确率
        acc_D_test = clients[i].test("test")
        time2 = time.time()
        acc_D_fs.append(acc_D_f)
        acc_D_rs.append(acc_D_r)
        acc_D_tests.append(acc_D_test)
        train_times.append(time2 - time1)

    acc_D_fs.append(np.mean(np.array(acc_D_fs)))
    acc_D_rs.append(np.mean(np.array(acc_D_rs)))
    acc_D_tests.append(np.mean(np.array(acc_D_tests)))
    train_times.append(np.mean(np.array(train_times)))

    df = pd.DataFrame(
        {
            "D_f": acc_D_fs,
            "D_r": acc_D_rs,
            "D_test": acc_D_tests,
            "train_time": train_times,
        }
    )

    if not os.path.exists(model_save_path()):
        os.makedirs(model_save_path())
    df.to_csv(
        path.join(model_save_path(), f"unlearn_{lam}_results.csv"),
        index=True,
        float_format="%.4f",
    )
