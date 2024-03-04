import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
import os
from utils.color import colors
from os import path
from utils.get_path import *
import seaborn as sns
sns.set_style("darkgrid")


def plt_result():
    idx = 0
    config = yaml.load(
        open(get_config_path()),
        Loader=yaml.FullLoader,
    )

    users = [i for i in range(config["clients"])]

    plt.figure(figsize=(6, 4))
    graph_path = graph_save_path()

    df = pd.read_csv(path.join(model_save_path(), "target_train_results.csv"))

    plt.plot(
        users,
        np.array(df["D_test"])[:-1],
        label=f"target avg:{np.array(df['D_test'])[-1]}",
        c=colors[idx],
    )
    idx += 1

    df = pd.read_csv(path.join(model_save_path(), "retrain_train_results.csv"))

    plt.plot(
        users,
        np.array(df["D_test"])[:-1],
        label=f"retrain avg:{np.array(df['D_test'])[-1]}",
        c=colors[idx],
    )
    idx += 1

    for lam in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
        df = pd.read_csv(path.join(model_save_path(), f"unlearn_{lam}_results.csv"))
        plt.plot(
            users,
            np.array(df["D_test"])[:-1],
            label=f"{lam} avg:{np.array(df['D_test'])[-1]}",
            lw=1,
            c=colors[idx],
        )
        idx += 1

    plt.legend()
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
        
        
    plt.xticks(users)
    plt.yticks(np.arange(0.5, 1, 0.1))
    plt.xlabel("Client ID")
    plt.ylabel("Test accuracy")
    # plt.grid()
    plt.savefig(
        f"{graph_path}/unlearn_result.pdf",
        format="pdf",
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()

