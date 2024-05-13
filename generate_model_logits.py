from resnet import ResNet
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="models/resnet_1.pth")
parser.add_argument("--batch_size", default=1000)
parser.add_argument("--save_dir", default="saves/logits")
args = vars(parser.parse_args())


for k in range(0, 25):
    print(k)
    resnet = ResNet()
    args["model_path"] = f"models/resnet_{k}.pth"

    weights_path = args["model_path"]
    model_name = args["model_path"].split("/")[-1].split(".")[0]
    model_idx = model_name.split("_")[-1]
    resnet.load_parameters(weights_path)
    acc, loss = resnet.test(model_idx)
    print(f"Test Acc: {acc}, Test loss: {loss}")
    resnet.net.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
    dataset_train = datasets.CIFAR10('data/cifar', download=True, transform=transform, train=True)
    dataset_test = datasets.CIFAR10('data/cifar', download=True, transform=transform, train=False)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args["batch_size"], shuffle=False)


    # Inference whole dataset
    logits_arr_train = np.empty((0,10))
    targets_arr_train = np.empty((0))
    logits_arr_test = np.empty((0,10))
    targets_arr_test = np.empty((0))

    with torch.no_grad():
        for i, batch in enumerate(data_loader_train, 0):
            print(f"{i+1}/{len(data_loader_train)}")
            x, y = batch
            x = x.to(resnet.device)
            y = y.to(resnet.device)
            logits = resnet.net(x)

            targets_arr_train = np.concatenate([targets_arr_train,y.cpu().numpy()])
            logits_arr_train = np.concatenate([logits_arr_train,logits.cpu().numpy()])

    with torch.no_grad():
        for i, batch in enumerate(data_loader_test, 0):
            print(f"{i+1}/{len(data_loader_test)}")
            x, y = batch
            x = x.to(resnet.device)
            y = y.to(resnet.device)
            logits = resnet.net(x)

            targets_arr_test = np.concatenate([targets_arr_test,y.cpu().numpy()])
            logits_arr_test = np.concatenate([logits_arr_test,logits.cpu().numpy()])


    np.savez_compressed(f"{args['save_dir']}/logits_{model_name}",
                        targets_arr_test=targets_arr_test,
                        targets_arr_train=targets_arr_train,
                        logits_arr_train=logits_arr_train,
                        logits_arr_test=logits_arr_test)