import torch
import torch.nn as nn
from torch.autograd import Variable

if __name__ == '__main__':
    def Train(my_net, device, trainloader):
        optim = torch.optim.SGD(my_net.parameters(), lr=0.001, momentum=0.9)
        loss_function = nn.BCELoss()
        epoch_num = 100

        # train
        for epoch in range(epoch_num):
            for i, (inputs, labels) in enumerate(trainloader):
                print(i, "번째")
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optim.zero_grad()
                out = my_net(inputs)
                loss = loss_function(torch.squeeze(out), torch.squeeze(labels))
                loss.backward()
                optim.step()
                if i % 10 == 0:
                    print("%d=> loss : %.3f" % (i, loss))

        print("train over")
        # 모델 저장
        torch.save(my_net.state_dict(), './cnn.pth')

