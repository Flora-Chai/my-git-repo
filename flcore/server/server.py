from utils.model import *
import copy
import time
import yaml
import numpy as np
from utils.gen_target_model import *
from flcore.client.client import Client
from HyperPrams import get_parser
import matplotlib.pyplot as plt
from utils.get_path import *
from tqdm import tqdm

class Server:
    def __init__(self, model_type):
        self.args = get_parser()
        #self.model = LeNet5().to(self.args.device)
        self.model =TransformerModel().to(self.args.device)
        self.config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
        self.global_loss = []
        self.model_type = model_type

    from torch.nn import TransformerEncoderLayer

    # 在初始化 Server 类时传递必要的参数
    class Server:
        def __init__(self, model_type):
            self.args = get_parser()
            # 使用一个 Transformer 编码器层，num_blocks 设置为 2
            self.model = TransformerModel(block=TransformerEncoderLayer(d_model=32, nhead=4), num_blocks=2).to(
                self.args.device)
            self.config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
            self.global_loss = []
            self.model_type = model_type

    def fed_avg(self, clients):
        """
        Returns the average of the weights.
        """
        w = []
        for client in clients:
            w.append(copy.deepcopy(client.model.state_dict()))

        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        self.model.load_state_dict(copy.deepcopy(w_avg))

    def train(self):
        args = get_parser()
        time1 = time.time()
        clients = []
        for i in range(self.config["clients"]):
            clients.append(Client(i))

        for epoch in tqdm(range(self.config["server_epochs"])):
            # print(f"========================{epoch}========================")
            local_loss = []
            # random choice client to train
            m = max(int(self.config["frac"] * self.config["clients"]), 1)
            idxs_users = np.random.choice(
                range(self.config["clients"]), m, replace=False
            )
            train_clients = []
            for idx in idxs_users:
                train_clients.append(clients[idx])
            self.model.train()
            for client in train_clients:
                client.update_weight(self.model)
                # test_acc = client.test_client()
                # print(f"|\tclient{client.id} \t|\t{round(test_acc * 100, 2)}%\t\t|")
                if client.id == args.id and self.model_type == "retrain":
                   local_loss.append(client.train_client("rest"))
                else:
                    local_loss.append(client.train_client("train"))
            self.fed_avg(train_clients)
            self.global_loss.append(np.mean(local_loss))

        for i in range(len(clients)):
            clients[i].update_weight(self.model)
        time2 = time.time()
        generate_target_model(self, time2 - time1, self.model_type)

    def plt_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(0, self.config['server_epochs']), self.global_loss, label="loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"{model_save_path()}/{self.model_type}_loss.png", dpi=300)