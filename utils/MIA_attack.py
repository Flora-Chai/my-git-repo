from os import path
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.get_path import *
from utils.model import *
from utils.data.data_augment import data_aug
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def MIA_data(shadow_model, in_data, out_data, args, config):
    # very important！！！

    shadow_model.eval()

    """
        in_label 全1 标记为训练数据
        out_label 全0 标记为非训练数据
    """
    in_data, in_label = torch.from_numpy(in_data), torch.ones(size=(in_data.shape[0],))
    out_data, out_label = torch.from_numpy(out_data), torch.zeros(
        size=(out_data.shape[0],)
    )
    in_data = torch.tensor(in_data, dtype=torch.float32)
    out_data = torch.tensor(out_data, dtype=torch.float32)
    InData = data_aug((in_data, in_label), args, "test")
    OutData = data_aug((out_data, out_label), args, "test")

    Inloader = DataLoader(InData, batch_size=config["mia_batch_size"])
    Outloader = DataLoader(OutData, batch_size=config["mia_batch_size"])

    # 使用Din训练shadow_model
    attack_x = []
    attack_y = []
    for i, (x, y) in enumerate(tqdm(Inloader)):
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            pred = F.softmax(shadow_model(x))
            # 降序排序，输出预测
            pred, _ = torch.sort(pred, descending=True)
            # 取前三个概率最大的预测值
            pred = pred[:, :4]
        """
            pred是预测最大的三个
            y是全1
        """
        attack_x.append(pred.detach())
        attack_y.append(y.detach())

    # Dout
    for i, (x, y) in enumerate(tqdm(Outloader)):
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            pred = F.softmax(shadow_model(x))
            pred, _ = torch.sort(pred, descending=True)
            pred = pred[:, :4]
        attack_x.append(pred.detach())
        attack_y.append(y.detach())

    tensor_x = torch.cat(attack_x)
    tensor_y = torch.cat(attack_y)

    # return attackloader, attacktester
    data = tensor_x.detach().cpu().numpy()
    target = tensor_y.detach().cpu().numpy()

    return data, target


def train_attacker(data_x, data_y, args):
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    attack_model = XGBClassifier(
        n_estimators=800,
        n_jobs=4,
        max_depth=6,
        objective="binary:logistic",
        booster="gbtree",
    )
    # attack_model = XGBClassifier()
    attack_model.fit(X_train, y_train)
    if not path.exists(path.join(model_save_path(), f"{args.id}")):
        os.makedirs(path.join(model_save_path(), f"{args.id}"))
    attack_model.save_model(
        path.join(model_save_path(), f"{args.id}", "attack_model.json")
    )

    print("\n")
    print(
        "MIA Attacker training accuracy: {}".format(
            accuracy_score(y_train, attack_model.predict(X_train))
        )
    )
    print(
        "MIA Attacker testing accuracy: {}".format(
            accuracy_score(y_test, attack_model.predict(X_test))
        )
    )
    print("\n")
    df = pd.DataFrame({
        "type": ["train", "test"],
        "value": [accuracy_score(y_train, attack_model.predict(X_train)), accuracy_score(y_test, attack_model.predict(X_test))]
    })

    df.to_csv(
        f"{model_save_path()}/mia_train_result.csv",
        index=False,
        float_format="%.4f"
    )


def test_attacker(data_x, data_y, args):
    attack_model = XGBClassifier(
        n_estimators=800,
        n_jobs=4,
        max_depth=6,
        objective="binary:logistic",
        booster="gbtree",
    )
    # attack_model = XGBClassifier(n_jobs=4, objective='binary:logistic', booster="gbtree")
    attack_model.load_model(
        path.join(model_save_path(), f"{args.id}", "attack_model.json")
    )

    # print(classification_report(data_y, attack_model.predict(data_x)))
    accuracy = accuracy_score(data_y, attack_model.predict(data_x))
    # print(f'MIA accuracy: {accuracy}')

    return accuracy


def MIA():
    from utils.model import LeNet5,TransformerModel
    from HyperPrams import get_parser

    args = get_parser()
    config = yaml.load(
        open(get_config_path()),
        Loader=yaml.FullLoader,
    )
    # dataset
    """
        1训练数据
        2非训练数据
    """
    data_in = np.load(path.join(get_data_path(), f"{args.id}", "data_batch_0.npz"))["x"]
    data_out = np.load(path.join(get_data_path(), f"{args.id}", "data_batch_1.npz"))[
        "x"
    ]

    # shadow model
    #shadow_model = LeNet5().to(args.device)
    shadow_model = TransformerModel().to(args.device)
    shadow_model.load_state_dict(
        torch.load(
            path.join(model_save_path(), "target_model.pt"),
            #map_location=f"cuda:{args.gpu_id}"
        )
    )

    # construct the training data for MIA
    data_x, data_y = MIA_data(shadow_model, data_in, data_out, args, config)

    # train or test
    train_attacker(data_x, data_y, args)
    # test_attacker(data_x, data_y, args)
