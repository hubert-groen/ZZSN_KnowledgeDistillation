from resnet import ResNet
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument("--batch_size", required=False, default=1000, type=int)
parser.add_argument("--index", required=True, type=int)
parser.add_argument("--save_dir", required=True)
args = vars(parser.parse_args())

resnet = ResNet()

train_idx = args["index"]
weights_path = args["model_path"]
resnet.load_parameters(weights_path)
resnet.net.eval()

transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
dataset = datasets.CIFAR10('data/cifar', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)


# Inference whole dataset
logits_arr = np.empty((0,10))
targets_arr = np.empty((0))

with torch.no_grad():
    for i, batch in enumerate(data_loader, 0):
        print(f"{i+1}/{len(data_loader)}")
        x, y = batch
        x = x.to(resnet.device)
        y = y.to(resnet.device)
        logits = resnet.net(x)

        targets_arr = np.concatenate([targets_arr,y.cpu().numpy()])
        logits_arr = np.concatenate([logits_arr,logits.cpu().numpy()])


logits_arr = logits_arr.astype(np.float32)
targets_arr = logits_arr.astype(np.float32)
np.save(f"{args['save_dir']}/logits/logits_{train_idx}", logits_arr)
np.save(f"{args['save_dir']}/logits/targets_{train_idx}", targets_arr)