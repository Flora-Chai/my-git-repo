from flcore.client.client import Client
import torch
import torchvision.transforms as transforms
import numpy as np
from utils.data.data_augment import data_aug
from HyperPrams import get_parser

args = get_parser()

path = "result/MNIST/third_rest_10_5_128_16_16_200_0_0.1_0.01_0.5_reserved"
target_path = path + "/target_model.pt"
unlearn_path = path + "/0/unlearn_0.7_model.pt"
retrain_path = path + "/retrain_model.pt"

client = Client(0)
client.model.eval()
x = torch.from_numpy(np.load("./npz_data/MNIST/0/data_batch_0.npz")['x'][:10])
transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),
                ]
            )

x1 = torch.empty(size=(0, 1, 32, 32))
for i in range(10):
    x1 = torch.cat([x1, transform(x[i]).unsqueeze(0)],dim=0)
    
x1 = x1.to(client.device)

client.model.load_state_dict(torch.load(target_path))

print(f"==================target=================")
with torch.no_grad():
    output = client.model(x1)
    print(output)

client.model.load_state_dict(torch.load(unlearn_path))

print(f"==================unlearn=================")
with torch.no_grad():
    output = client.model(x1)
    print(output)

client.model.load_state_dict(torch.load(retrain_path))

print(f"==================retrain=================")
with torch.no_grad():
    output = client.model(x1)
    print(output)

