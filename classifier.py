import torchvision
from torchvision import datasets, transforms
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
from torch.autograd import Variable as var
import time


class MNIST_classifier2(nn.Module):
    def __init__(self):
        super(MNIST_classifier2, self).__init__()
        # Write the code defining the structure of the network here
        self.layer1=nn.Linear(28*28, 1000)
        self.layer2=nn.Linear(1000, 1000)
        self.layer3=nn.Linear(1000, 10)
        self.relu=nn.ReLU()
        
    def forward(self, x):
        # Define what a forward pass looks like here:
        # Erase the 'pass' keyword and insert your own code.
        x = images.view(-1,28*28)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x, dim=1)

batch_size = 64
n_epochs=1
model = MNIST_classifier2()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.SGD(params=params,lr=0.001)

training_set = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

t0=time.time()
for e in range(n_epochs):
    for i,(images,labels) in enumerate(training_set):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)
        model.zero_grad()
        loss = cec_loss(output,labels)
        loss.backward()
        optimizer.step()
        #print(loss.item())
t1=time.time()
trainingtime=t1-t0

test_batch_size = 1000
test_set = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

total = 0
correct = 0
for i,(images,labels) in enumerate(test_set):
    images = var(images.cpu())
    x = model(images)
    value,pred = torch.max(x,1)
    pred = pred.data.cpu()
    total += x.size(0)
    correct += torch.sum(pred == labels)

# What accuracy is achieved? 
print('Accuracty:')
print(correct.item()*100./total, end ="")
print('%')

# What is the size of the network (in MB)? 
summary(model, input_size=(1, 28, 28))

# What is the training time?
print('Training time for 1 epoch:')
print(trainingtime, end ="")
print ('s')
