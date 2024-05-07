import torch
import torchvision.transforms as transforms
from torch import nn
import torchvision.datasets as datasets
import os
import model
import util
import numpy as np
from torch.utils.data import Subset


class ResNet:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.net = model.Net().cuda() if self.use_cuda else model.Net()
        self.optimizer = None
        self.train_accuracies = []
        self.test_accuracies = []
        self.start_epoch = 1

    def train(self, save_dir, read_index, num_epochs=75, batch_size=256, learning_rate=0.001, test_each_epoch=False, verbose=False, teacher_logits_path=None):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.read_index = read_index
        self.net.train()

        train_transform = transforms.Compose([
            util.Cutout(num_cutouts=2, size=8, p=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform)
        train_idx = np.load(f'indices/train_idx_{read_index}.npy')
        if teacher_logits_path is not None:
            teacher_logits = np.load(teacher_logits_path)
            train_dataset.targets = teacher_logits
        train_subset = Subset(train_dataset, train_idx)
        data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()

        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = 0
            epoch_total = 0
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                if teacher_logits_path is None:
                    loss = criterion(outputs, labels.squeeze_())
                else:
                    soft_targets = nn.functional.softmax(labels, dim=-1)
                    soft_prob = nn.functional.log_softmax(outputs, dim=-1)
                    loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0]
                loss.backward()
                self.optimizer.step()

                if teacher_logits_path is None:
                    _, predicted = torch.max(outputs.data, dim=1)
                    batch_total = labels.size(0)
                    batch_correct = (predicted == labels.flatten()).sum().item()

                    epoch_total += batch_total
                    epoch_correct += batch_correct

                    if verbose:
                        # Update progress bar in console
                        info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                    format(batch_correct / batch_total, epoch_correct / epoch_total)
                        progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)
                        self.train_accuracies.append(epoch_correct / epoch_total)

                else:
                    if verbose:
                        # Update progress bar in console
                        info_str = 'Loss: {:.4f}'.\
                                    format(loss)
                        progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)           

            if verbose:
                progress_bar.new_line()

            if test_each_epoch:
                test_accuracy = self.test()
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every epoch
            self.save_parameters(epoch, directory=save_dir)

    def test(self, read_index, batch_size=256):
        """Tests the network.

        """
        self.net.eval()

        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                             ])

        test_dataset = datasets.CIFAR10('data/cifar', download=True, transform=test_transform, train=False)
        train_idx = np.load(f'indices/test_idx_{read_index}.npy')
        train_subset = Subset(test_dataset, train_idx)
        data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum().item()

        self.net.train()
        return correct / total

    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }, os.path.join(directory, 'resnet_' + str(self.read_index) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        self.optimizer = torch.optim.Adam(self.net.parameters())
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        self.start_epoch = checkpoint['epoch']

