#!/usr/bin/env python3

import torch
import torchvision
import sklearn.metrics as metrics
import numpy as np
import sys
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split
from config import device

import model

# This class allows train/test split with different transforms
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Test network on validation set, if it exists
def test_network(net,testloader,print_confusion=False):
    net.eval()
    total_images = 0
    total_correct = 0
    conf_matrix = np.zeros((151,151))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                labels.cpu(),predicted.cpu(),labels=list(range(151)))

    model_accuracy = total_correct / total_images * 100
    print(', {0} test {1:.2f}%'.format(total_images,model_accuracy))
    if print_confusion:
        np.set_printoptions(precision=2, suppress=True)
        print(conf_matrix)
    net.train()
    return model_accuracy

# Plot train (and test, if it exists) accuracy periodically per epoch
def plot_accuracy(train_acc_history, test_acc_history, filename='accuracy_plot.png'):
    epochs_range = range(1, len(train_acc_history) + 1)
    plt.figure()
    plt.plot(epochs_range, train_acc_history, label='Train accuracy')
    if test_acc_history:
        plt.plot(epochs_range, test_acc_history, label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs test accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.show()
    print('   Accuracy chart saved to {0}'.format(filename))

def main():
    print("Using device: {}"
          "\n".format(str(device)))
    ########################################################################
    #######                      Loading Data                        #######
    ########################################################################
    data = torchvision.datasets.ImageFolder(root=model.dataset)
    
    if model.train_val_split == 1:
        # Train on the entire dataset
        data = torchvision.datasets.ImageFolder(root=model.dataset,
                            transform=model.transform('train'))
        trainloader = torch.utils.data.DataLoader(data,
                            batch_size=model.batch_size, shuffle=True);
    else:
        # Split the dataset into trainset and testset
        data = torchvision.datasets.ImageFolder(root=model.dataset)
        data.len=len(data)
        train_len = int((model.train_val_split)*data.len)
        test_len = data.len - train_len
        train_subset, test_subset = random_split(data, [train_len, test_len])
        trainset = DatasetFromSubset(
            train_subset, transform=model.transform('train'))
        testset = DatasetFromSubset(
            test_subset, transform=model.transform('test'))

        train_sampler = model.make_train_sampler(train_subset)
        trainloader = torch.utils.data.DataLoader(trainset,
                            batch_size=model.batch_size, sampler=train_sampler)
        testloader = torch.utils.data.DataLoader(testset,
                            batch_size=model.batch_size, shuffle=False)

    # Get model, loss criterion and optimizer from model
    net = model.net.to(device)
    criterion = model.loss_func
    optimizer = model.optimizer
    # get weight initialization and lr scheduler, if appropriate
    weights_init = model.weights_init
    scheduler = model.scheduler

    # apply custom weight initialization, if it exists
    net.apply(weights_init)
    
    ########################################################################
    #######                        Training                          #######
    ########################################################################
    print("Start training...")
    train_acc_history = []
    test_acc_history = []
    for epoch in range(1,model.epochs+1):
        total_loss = 0
        total_images = 0
        total_correct = 0

        for batch in trainloader:           # Load batch
            images, labels = batch 
            images = images.to(device)
            labels = labels.to(device)

            preds = net(images)             # Process batch
            
            loss = criterion(preds, labels) # Calculate loss

            optimizer.zero_grad()
            loss.backward()                 # Calculate gradients
            optimizer.step()                # Update weights

            output = preds.argmax(dim=1)

            total_loss += loss.item()
            total_images += labels.size(0)
            total_correct += output.eq(labels).sum().item()

        # apply lr schedule, if it exists
        if scheduler is not None:
            scheduler.step()
            
        model_accuracy = total_correct / total_images * 100
        train_acc_history.append(model_accuracy)
        print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
               epoch, total_loss, total_images, model_accuracy), end='')

        if model.train_val_split < 1:
            test_acc = test_network(net,testloader,
                         print_confusion=(epoch % 10 == 0))
            test_acc_history.append(test_acc)
        else:
            print()

        if epoch % 10 == 0:
            torch.save(net.state_dict(),'checkModel.pth')
            print("   Model saved to checkModel.pth")
            plot_accuracy(train_acc_history, test_acc_history)

        sys.stdout.flush()

    torch.save(net.state_dict(),'savedModel.pth')
    print("   Model saved to savedModel.pth")
    plot_accuracy(train_acc_history, test_acc_history)
        
if __name__ == '__main__':
    main()
