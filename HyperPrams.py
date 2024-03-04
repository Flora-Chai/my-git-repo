import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--model_type", type=str, default="target")
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    # 第几个客户端是要忘记的
    parser.add_argument(
        "--id", type=int, default=0, help="which client that you will forgotten"
    )
    parser.add_argument("--forget_size", type=int, default=200)

    args = parser.parse_args()
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    args.device = device

    return args
