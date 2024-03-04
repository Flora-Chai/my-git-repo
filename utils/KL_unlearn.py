import numpy as np
import torch.nn as nn
import torch
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from utils.model import LeNet5, TransformerModel
from utils.train_test_model import test
import time
from utils.MIA_attack import test_attacker, MIA_data
from utils.data.data_utils import read_client_data
from torch.utils.data import DataLoader
from os import path
from utils.get_path import *
import matplotlib.pyplot as plt
from utils.color import colors

import warnings

warnings.filterwarnings("ignore")


def Unlearn(args):
    config = yaml.load(
        open(get_config_path()),
        Loader=yaml.FullLoader,
    )

    torch.manual_seed(args.seed)
    trained_model_path = path.join(model_save_path(), "target_model.pt")
    testloader = DataLoader(
        read_client_data(args.id, "test"),
        batch_size=config["test_batch_size"],
        shuffle=True,
        drop_last=True,
    )
    forgetloader = DataLoader(
        read_client_data(args.id, "forgotten"),
        batch_size=config["forget_batch_size"],
        shuffle=True,
        drop_last=True,
    )
    thirdloader = DataLoader(
        read_client_data(args.id, config["forget_data"]),
        batch_size=config["forget_batch_size"],
        shuffle=True,
        drop_last=True,
    )
    balanceloader = DataLoader(
        read_client_data(args.id, config["balance_data"]),
        batch_size=config["balance_batch_size"],
        shuffle=True,
        drop_last=True,
    )

    #trained_model = LeNet5().to(args.device)
    trained_model = TransformerModel().to(args.device)
    trained_model.load_state_dict(torch.load(trained_model_path))
    # generator
    #generator = LeNet5().to(args.device)
    generator = TransformerModel().to(args.device)
    generator.load_state_dict(torch.load(trained_model_path))

    # Optimizers
    optimizer_G = torch.optim.SGD(generator.parameters(), config["ur"])

    # KL散度损失
    criterion_forget = nn.KLDivLoss()
    # 交叉损失
    criterion_performance = nn.CrossEntropyLoss()

    loss1 = []
    loss2 = []
    loss3 = []
    total_time = []

    data_in = np.load(path.join(get_data_path(), f"{args.id}", "data_batch_0.npz"))[
        "x"
    ][: args.forget_size]
    data_out = np.load(path.join(get_data_path(), f"{args.id}", "data_batch_2.npz"))[
        "x"
    ][: args.forget_size]

    accBefore_D_test = test(generator, testloader, args)

    data_x, data_y = MIA_data(generator, data_in, data_out, args, config)
    accBefore_MIA = test_attacker(data_x, data_y, args)
    print(f"accuracy of D_test: {accBefore_D_test}")
    print(f"accuracy of MIA: {accBefore_MIA}")

    from utils.pytorchtools import EarlyStopping

    early_stopping = EarlyStopping(
        patience=7,
        delta=0.01,
        path=path.join(
            model_save_path(), f"{args.id}", f"unlearn_{args.Lambda}_model.pt"
        ),
    )

    trained_model.eval()
    generator.train()
    for epoch in tqdm(range(config["unlearn_epochs"])):
        epoch_loss1 = []
        epoch_loss2 = []
        epoch_loss3 = []
        for idx, (BA_data, _) in enumerate(forgetloader):
            time_1 = time.time()

            # the forgotten data
            BA_data = BA_data.to(args.device)

            # the third-party data
            # TD_data, _ = next(iter(thirdloader))
            TD_data = torch.normal(0, 1, size=BA_data.shape)
            TD_data = TD_data.to(args.device)

            # the rest data to maintain the performance
            (RD_data, RD_label) = next(iter(balanceloader))
            RD_data, RD_label = RD_data.to(args.device), RD_label.to(args.device)

            # output of fixed model for the TD_data
            # 训练模型在第三方数据下的输出
            with torch.no_grad():
                O_t = F.softmax(trained_model(TD_data))

            optimizer_G.zero_grad()

            # output of generator for the forgotten data
            # 遗忘数据在generator下的输出
            O_f = F.log_softmax(generator(BA_data))

            # 𝜆𝐾𝐿(Pr(𝑀𝑢(𝐷𝑓 ))∥Pr(𝑀𝑡(𝐷𝑡))), 遗忘数据和第三方数据在模型输出下的KL散度
            loss_forget = args.Lambda * criterion_forget(O_f, O_t)

            # (1 − 𝜆)(𝑀𝑢(𝐷𝑟)) 权衡模型在遗忘和剩余数据的表现
            loss_performance = (1 - args.Lambda) * criterion_performance(
                generator(RD_data), RD_label
            )

            # g_loss = (loss_forget + loss_performance) / config["clients"]
            g_loss = loss_forget + loss_performance

            epoch_loss1.append(loss_forget.item())
            epoch_loss2.append(loss_performance.item())
            epoch_loss3.append(g_loss.item())

            # 反向传播
            g_loss.backward()
            optimizer_G.step()

            time_2 = time.time()
            total_time.append(time_2 - time_1)

            early_stopping(g_loss.item(), generator)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        loss1.append(np.mean(epoch_loss1))
        loss2.append(np.mean(epoch_loss2))
        loss3.append(np.mean(epoch_loss3))

    if not path.exists(f"{model_save_path()}/{args.Lambda}"):
        os.makedirs(f"{model_save_path()}/{args.Lambda}")

    plt.figure()
    plt.plot([i for i in range(len(loss1))], loss1, c=colors[0])
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("forget loss")
    plt.savefig(f"{model_save_path()}/{args.Lambda}/forget_loss.png", dpi=300)

    plt.figure()
    plt.plot([i for i in range(len(loss1))], loss1, c=colors[0])
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("performance loss")
    plt.savefig(f"{model_save_path()}/{args.Lambda}/performance_loss.png", dpi=300)

    plt.figure()
    plt.plot([i for i in range(len(loss1))], loss1, c=colors[0])
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("g loss")
    plt.savefig(f"{model_save_path()}/{args.Lambda}/g_loss.png", dpi=300)

    total_time = np.sum(total_time)
    print(f"total time: {total_time:.2f}s")

    # 遗忘后的测试精度
    accAfter_D_test = test(generator, testloader, args)
    data_x, data_y = MIA_data(generator, data_in, data_out, args, config)
    accAfter_MIA = test_attacker(data_x, data_y, args)
    print(f"accuracy of D_test: {accAfter_D_test}")
    print(f"accuracy of MIA: {accAfter_MIA}")

    return accBefore_D_test, accAfter_D_test, accBefore_MIA, accAfter_MIA, total_time
