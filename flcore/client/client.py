import copy

import numpy as np
import yaml
from torch.utils.data import DataLoader
from utils.data.data_utils import read_client_data
from utils.model import *
import pandas as pd
from HyperPrams import get_parser
from utils.KL_unlearn import Unlearn
from os import path
from utils.get_path import *


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, id):
        self.args = get_parser()
        self.device = self.args.device
        #self.model = LeNet5().to(self.device)
        self.model = TransformerModel().to(self.device)
        self.id = id  # integer
        self.config = yaml.load(
            open(get_config_path()),
            Loader=yaml.FullLoader,
        )

        self.learning_rate = self.config["rate"]
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate
        )  # momentum=0.9, weight_decay=1e-4
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=0.99
        )
        self.learning_rate_decay = True
        self.future_test = False

    # load client data
    def load_data(self, batch_size=None, data_type=None):
        data = read_client_data(self.id, data_type)
        return DataLoader(data, batch_size, drop_last=True, shuffle=True)

    # update client weight
    def update_weight(self, model):
        self.model.load_state_dict(copy.deepcopy(model.state_dict()))

    def train_client(self, data_type="train"):
        epochs = self.config["client_epochs"]
        # if self.id == 0 or self.id == 1:
        #     epochs += 10
        loss_arr = []
        for i in range(epochs):
            trainloader = self.load_data(self.config["train_batch_size"], data_type)
            self.model.train()

            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss_arr.append(loss.item())
                loss.backward()
                self.optimizer.step()
        return np.mean(loss_arr)

    def test_client(self):
        res_acc = []
        for i in range(self.config["test_epochs"]):
            testloaderfull = self.load_data(self.config["test_batch_size"], "test")
            self.model.eval()
            test_acc = 0
            test_num = 0
            with torch.no_grad():
                for x, y in testloaderfull:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)

                    test_acc += (output.argmax(1) == y).sum().item()
                    test_num += y.shape[0]
            if test_num == 0:
                continue
            else:
                res_acc.append(test_acc / test_num)
        return np.mean(res_acc)

    def test(self, data_type):
        res_acc = []
        for _ in range(self.config["test_epochs"]):
            testloaderfull = self.load_data(self.config["test_batch_size"], data_type)
            self.model.eval()
            test_acc = 0
            test_num = 0
            with torch.no_grad():
                for x, y in testloaderfull:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)
                    pred, _ = torch.sort(output, descending=True)
                    test_acc += (output.argmax(1) == y).sum().item()
                    test_num += y.shape[0]

            if test_num == 0:
                continue
            else:
                res_acc.append(test_acc / test_num)
        if len(res_acc) == 0:
            return 0
        return np.mean(res_acc)

#FU 遗忘某个client
    def kl_unlearn(self):
        # 参数解析
        # 测试不同的λ
        # for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # for lam in [0.01]:
        for lam in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
            self.args.Lambda = lam
            (
                accBefore_D_test,
                accAfter_D_test,
                accBefore_MIA,
                accAfter_MIA,
                total_time,
            ) = Unlearn(self.args)

            # 将参数和返回值添加到DataFrame中
            results_df = pd.DataFrame(
                {
                    "name": [
                        "lambda",
                        "accBefore_D_test",
                        "accAfter_D_test",
                        "accBefore_MIA",
                        "accAfter_MIA",
                        "total_time",
                    ],
                    "result": [
                        lam,
                        accBefore_D_test,
                        accAfter_D_test,
                        accBefore_MIA,
                        accAfter_MIA,
                        total_time,
                    ],
                }
            )

            print("saving.....")
            results_df.to_csv(
                path.join(
                    model_save_path(), f"{self.args.id}", f"unlearn_{lam}_results.csv"
                ),
                index=False,
                float_format="%.4f",
            )
