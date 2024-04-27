from resnet import ResNet

index = 9

net = ResNet()
net.load_parameters(path='saves/resnet__epoch_75.pth')
net.train(save_dir='saves', read_index=index, num_epochs=75, batch_size=256, learning_rate=0.001, verbose=True)
accuracy = net.test(index=9)
print('Test accuracy: {}'.format(accuracy))
