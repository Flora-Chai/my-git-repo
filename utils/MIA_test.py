from utils.model import *
from HyperPrams import get_parser
import numpy as np
from utils.get_path import *
import yaml
from utils.MIA_attack import *
import pandas as pd

def get_mia_result():
    # 获取命令行参数和配置信息
    args = get_parser()
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)

    # 设置目标模型路径
    target_model_path = f"{model_save_path()}/target_model.pt"

    # 创建LeNet5模型对象并加载目标模型权重
    #model = LeNet5().to(args.device)
    model = TransformerModel().to(args.device)
    model.load_state_dict(torch.load(target_model_path))

    # 获取训练和非训练数据
    data_in = np.load(path.join(get_data_path(), f"{args.id}", "data_batch_0.npz"))["x"][: args.forget_size]
    data_out = np.load(path.join(get_data_path(), f"{args.id}", "data_batch_2.npz"))["x"][: args.forget_size]

    # 生成MIA攻击数据并计算准确率
    data_x, data_y = MIA_data(model, data_in, data_out, args, config)
    target_MIA = test_attacker(data_x, data_y, args)

    # 设置重新训练模型路径
    retrain_model_path = f"{model_save_path()}/retrain_model.pt"
    # 加载重新训练模型权重
    model.load_state_dict(torch.load(retrain_model_path))
    # 生成MIA攻击数据并计算准确率
    data_x, data_y = MIA_data(model, data_in, data_out, args, config)
    retrain_MIA = test_attacker(data_x, data_y, args)

    # 将结果保存为CSV文件
    df = pd.DataFrame({
        "type": ["target", "retrain"],
        "value": [target_MIA, retrain_MIA]
    })

    df.to_csv(
        f"{model_save_path()}/mia_result.csv",
        index=False,
        float_format="%.4f"
    )
