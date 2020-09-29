import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model import Net

if __name__ == '__main__':
    # gpu 사용
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    trainInput = np.load('./dataset/x_train.npy')
    trainOutput = np.load('./dataset/y_train.npy')
    testInput = np.load('./dataset/x_val.npy')
    testOutput = np.load('./dataset/y_val.npy')

    # reshape
    trainInput = np.transpose(trainInput, (0, 3, 1, 2))
    trainOutput = np.transpose(trainOutput, (0, 3, 1, 2))
    testInput = np.transpose(testInput, (0, 3, 1, 2))
    testOutput = np.transpose(testOutput, (0, 3, 1, 2))

    # change numpy to tensor
    trainInput = torch.from_numpy(trainInput)
    trainOutput = torch.from_numpy(trainOutput)
    testInput = torch.from_numpy(testInput)
    testOutput = torch.from_numpy(testOutput)

    print("trainInput", trainInput.shape)
    print("trainOutput", trainOutput.shape)

    # make dataset
    trainset = TensorDataset(trainInput, trainOutput)
    testset = TensorDataset(testInput, testOutput)

    # make dataloader
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

    my_net = Net()
    my_net.load_state_dict(torch.load('./cnn.pth'))
    my_net.to(device)

    def imshow(img1, img2, img3):
        np_img1 = np.squeeze(img1, axis=0)
        np_img1 = np.squeeze(np_img1, axis=0)
        np_img2 = np.squeeze(img2, axis=0)
        np_img2 = np.squeeze(np_img2, axis=0)
        np_img3 = np.squeeze(img3, axis=0)
        np_img3 = np.squeeze(np_img3, axis=0)

        plt.subplot(1, 3, 1)
        plt.imshow(np_img1)
        plt.subplot(1, 3, 2)
        plt.imshow(np_img2)
        plt.subplot(1, 3, 3)
        plt.imshow(np_img3)

        plt.show()

    #test
    for i, (inputs, labels) in enumerate(testloader):
        if(i == 5):
            break
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = my_net(inputs)
        out =  out.detach().numpy()
        inputs = inputs.detach().numpy()
        labels = labels.detach().numpy()

        imshow(inputs, labels, out)




