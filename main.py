import os.path

import yaml

from flcore.server.server import *
from flcore.client.client import *
from utils.data.data_utils import *
from HyperPrams import get_parser
import time
from utils.gen_target_model import generate_target_model
from utils.MIA_attack import MIA
from utils.result import result_unlearn
from utils.graph import plt_result
import numpy as np
import random
from utils.get_path import *
from utils.MIA_test import get_mia_result


if __name__ == "__main__":
    args = get_parser()
    np.random.seed(1)
    random.seed(1)

    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)

    #
    # plt_result()
    if args.model_type == "target":
        args = get_parser()
        if not os.path.exists(os.path.join(get_data_path(), f"0/data_batch_0.npz")):
            data_split()

        server = Server("target")

        server.train()
        server.plt_loss()
        MIA()
        print(f"client {args.id} unlearning........")
        client = Client(args.id)
        client.kl_unlearn()

        print(f"unlearned testing ............")
        for lam in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
            result_unlearn(args, lam)
        print("retraining ........")

        # 重新初始化客户端retrain
        server = Server("retrain")

        server.train()
        server.plt_loss()
        plt_result()
        get_mia_result()
        
