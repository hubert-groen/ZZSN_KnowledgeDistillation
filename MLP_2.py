import torch
import torchvision.transforms as transforms
from torch import nn
import torchvision.datasets as datasets
import os
import util
import numpy as np
from torch.utils.data import Subset, ConcatDataset
import matplotlib.pyplot as plt
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MLPTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.mlp = None
        self.optimizer = None
        self.criterion = None
        self.train_accuracies = []
        self.test_accuracies = []
        self.start_epoch = 1

    def train_student(self, save_dir, read_index, num_epochs=75, batch_size=256, learning_rate=0.001, test_epoch=1, verbose=False):
        input_size = 32 * 32 * 3  # Assuming CIFAR-10 dataset
        hidden_size = 512  # You can adjust this
        num_classes = 10  # CIFAR-10 has 10 classes

        self.mlp = MLP(input_size, hidden_size, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

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

        progress_bar = util.ProgressBar()

        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = 0
            epoch_total = 0
            epoch_total_loss_train = 0

            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.mlp(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct
                epoch_total_loss_train += loss.item()

                if verbose:
                    info_str = f"Epoch accuracy: {batch_correct / batch_total}, Epoch loss: {loss.item()}"
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            epoch_accuracy = epoch_correct / epoch_total
            epoch_loss = epoch_total_loss_train / len(data_loader)
            self.train_accuracies.append(epoch_accuracy)
            if verbose:
                progress_bar.new_line()
                print(f"Epoch {epoch} accuracy: {epoch_accuracy}, Epoch loss: {epoch_loss}")

            if epoch % test_epoch == 0:
                test_accuracy = self.test(read_index)
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every epoch
            self.save_parameters(epoch, directory=save_dir)

    def train(self, save_dir, read_index, num_epochs=75, batch_size=256, learning_rate=0.001, test_epoch=1, verbose=False):
        """Trains the MLP network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        read_index : int
            Index used for loading data
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_epoch : int
            Test the network after every test_epoch epochs
        verbose : boolean
            Print training progress to console if True
        """
        input_size = 32 * 32 * 3  # Assuming CIFAR-10 dataset
        hidden_size = 512  # You can adjust this
        num_classes = 10  # CIFAR-10 has 10 classes

        self.mlp = MLP(input_size, hidden_size, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.read_index = read_index

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
                images = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.mlp(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct
                epoch_total_loss_train += loss.item()

                if verbose:
                    info_str = f"Epoch accuracy: {batch_correct / batch_total}, Epoch loss: {loss.item()}"
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            epoch_accuracy = epoch_correct / epoch_total
            epoch_accuracies_train.append(epoch_accuracy)
            epoch_loss = epoch_total_loss_train / len(data_loader)
            epoch_losses_train.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)
            if verbose:
                progress_bar.new_line()
                print(f"Epoch {epoch} accuracy: {epoch_accuracy}, Epoch loss: {epoch_loss}")

            if epoch % test_epoch == 0:
                test_accuracy, test_loss = self.test(read_index)
                epoch_losses_test.append(test_loss)
                epoch_accuracies_test.append(test_accuracy)
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))
                    print('Test loss: {}'.format(test_loss))

            # Save parameters after every epoch
            self.save_parameters(epoch_accuracies_train, epoch_accuracies_test, epoch_losses_train, epoch_losses_test, test_epoch, directory=save_dir)

        self.plot_metrics(epoch_losses_train, epoch_losses_test, epoch_accuracies_train, epoch_accuracies_test, read_index, test_epoch)
        


    def test(self, read_index, batch_size=1024):
        self.mlp.eval()

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10('data/cifar', transform=train_transform, train=True)
        test_dataset = datasets.CIFAR10('data/cifar', transform=train_transform, train=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        test_idx = np.load(f'indices/test_idx_{read_index}.npy')
        test_subset = Subset(full_dataset, test_idx)
        data_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        total_loss = 0
        loss_function = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                outputs = self.mlp(images)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        average_loss = total_loss / len(data_loader)

        self.mlp.train()
        return accuracy, average_loss

    def save_parameters(self, acc_train, acc_test, loss_train, loss_test, test_epoch, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({           
            'model_state_dict': self.mlp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracies_train': acc_train,
            'accuracies_test': acc_test,
            'losses_train': loss_train,
            'losses_test': loss_test,
            'test_epoch': test_epoch
        }, os.path.join(directory, 'mlp_' + str(self.read_index) + '.pth'))

    def load_parameters(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mlp.load_state_dict(checkpoint['model_state_dict'])

    def plot_metrics(self, epoch_losses_train, epoch_losses_test, epoch_accuracies_train, epoch_accuracies_test, read_index, test_epoch):
        epochs = len(epoch_losses_train)
        test_x_axis_ticks = np.arange(test_epoch, epochs+1, test_epoch)
        train_x_axis_ticks = np.arange(1, epochs+1, 1)
        save_dir='saves-MLP/plots'

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