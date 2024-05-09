import torch
import torchvision.transforms as transforms
from torch import nn
import torchvision.datasets as datasets
import os
import model
import util
import numpy as np
from torch.utils.data import Subset, ConcatDataset
import matplotlib.pyplot as plt


class ResNet:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.net = model.Net().cuda() if self.use_cuda else model.Net()
        self.optimizer = None
        self.train_accuracies = []
        self.test_accuracies = []
        self.start_epoch = 1


    def train_student(self, save_dir, read_index,teacher_logits_path, num_epochs=75, batch_size=256, learning_rate=0.001, test_epoch=1, verbose=False):
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

        train_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform, train=True)
        test_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform, train=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_idx = np.load(f'indices/train_idx_{read_index}.npy')
        teacher_logits = np.load(teacher_logits_path)
        full_dataset.targets = [(teacher_logit, target) for teacher_logit, target in zip(teacher_logits, full_dataset.targets)]
        train_subset = Subset(full_dataset, train_idx)
        data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        progress_bar = util.ProgressBar()

        epoch_accuracies_train = []
        epoch_accuracies_test = []
        epoch_losses_train = []
        epoch_losses_test = []

        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = 0
            epoch_total = 0
            epoch_total_loss_train = 0
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                logits, targets = labels
                logits = logits.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                soft_targets = nn.functional.softmax(logits, dim=-1)
                soft_prob = nn.functional.log_softmax(outputs, dim=-1)
                loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0]
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == targets.flatten()).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct
                epoch_total_loss_train += loss

            epoch_losses_train.append(epoch_total_loss_train/len(data_loader)) # sum of losses in each batch / total batches 
            epoch_accuracies_train.append(epoch_correct / epoch_total)

            if verbose:
                info_str = f"Epoch accuracy: {epoch_accuracies_train[-1]}, Epoch loss: {epoch_losses_train[-1]}"
                progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            if verbose:
                progress_bar.new_line()

            if epoch % test_epoch == 0:
                test_accuracy, test_loss = self.test(read_index=self.read_index)
                epoch_losses_test.append(test_loss)
                epoch_accuracies_test.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))
                    print('Test loss: {}'.format(test_loss))

            # Save parameters after every epoch
            self.save_parameters(epoch, directory=save_dir)
        
        self.plot_metrics(epoch_losses_train, epoch_losses_test, epoch_accuracies_train, epoch_accuracies_test, read_index, test_epoch)
    def train(self, save_dir, read_index, num_epochs=75, batch_size=256, learning_rate=0.001, test_epoch=1, verbose=False):
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
        test_epoch : int
            True: Test the network after every test_epoch epochs
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

        train_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform, train=True)
        test_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform, train=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_idx = np.load(f'indices/train_idx_{read_index}.npy')
        train_subset = Subset(full_dataset, train_idx)
        data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()

        epoch_accuracies_train = []
        epoch_accuracies_test = []
        epoch_losses_train = []
        epoch_losses_test = []

        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = 0
            epoch_total = 0
            epoch_total_loss_train = 0
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                loss = criterion(outputs, labels.squeeze_())
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels.flatten()).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct
                epoch_total_loss_train += loss

                if verbose:
                    info_str = f"Epoch accuracy: {epoch_accuracies_train[-1]}, Epoch loss: {epoch_losses_train[-1]}"
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            if verbose:
                progress_bar.new_line()

            if epoch % test_epoch == 0:
                test_accuracy, test_loss = self.test(read_index=self.read_index)
                epoch_losses_test.append(test_loss)
                epoch_accuracies_test.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))
                    print('Test loss: {}'.format(test_loss))

            # Save parameters after every epoch
            self.save_parameters(epoch, directory=save_dir)
        
        self.plot_metrics(epoch_losses_train, epoch_losses_test, epoch_accuracies_train, epoch_accuracies_test, read_index, test_epoch)

    def test(self, read_index, batch_size=256):
        """Tests the network.

        """
        self.net.eval()

        train_transform = transforms.Compose([
            util.Cutout(num_cutouts=2, size=8, p=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform, train=True)
        test_dataset = datasets.CIFAR10('data/cifar', download=True, transform=train_transform, train=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_idx = np.load(f'indices/test_idx_{read_index}.npy')
        train_subset = Subset(full_dataset, train_idx)
        data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        total_loss = 0
        loss_function = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = loss_function(outputs, labels)
                total_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum().item()

        accuracy = correct / total * 100
        average_loss = total_loss / total

        self.net.train()
        return accuracy, average_loss

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
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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

    def plot_metrics(self, epoch_losses_train, epoch_losses_test, epoch_accuracies_train, epoch_accuracies_test, read_index, test_epoch):
        epochs = len(epoch_losses_train)
        test_x_axis_ticks = np.arange(test_epoch, epochs, test_epoch)
        train_x_axis_ticks = np.arange(1, epochs, 1)
        save_dir='saves/plots'

        plt.figure(figsize=(10, 5))
        plt.style.use("ggplot")

        plt.subplot(1, 2, 1)
        plt.plot(train_x_axis_ticks, epoch_accuracies_train, label='Accuracy Train')
        plt.plot(test_x_axis_ticks, epoch_accuracies_test, label="Accuracy Test")
        plt.title('Epoch Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_x_axis_ticks, epoch_losses_train, label='Loss Train')
        plt.plot(test_x_axis_ticks, epoch_losses_test, label="Loss Test")
        plt.title('Epoch Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.suptitle(f'STATS for index = {read_index}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'accuracy-loss_{read_index}.png'))
        plt.close()