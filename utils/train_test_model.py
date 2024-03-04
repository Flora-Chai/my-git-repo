import torch
import numpy as np
import random
from utils.get_path import get_config_path
import yaml


def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def test(net, testloader, args):
    config = yaml.load(
        open(get_config_path()),
        Loader=yaml.FullLoader,
    )
    res_acc = []
    for i in range(config["test_epochs"]):
        net.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(args.device)
                y = y.to(args.device)
                output = net(x)

                test_acc += (output.argmax(1) == y).sum().item()
                test_num += y.shape[0]

        if test_num == 0:
            continue
        else:
            res_acc.append(test_acc / test_num)
    return np.mean(res_acc)
