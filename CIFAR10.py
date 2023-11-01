import torch
from torchvision import datasets, transforms

'''
Apply Batch Normalization to a simple CNN
to observe its effects on training 
'''



transform = transforms.Compose([
    transforms.toTensor(),
    transforms.Normalize((),())
])

# download the CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True,transform=transform) 
testset = datasets.CIFAR(root='/data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=True)

class CNNwoBN(torch.nn.Module):
    def __init__():
        #2 conv. layers
        #2 connected layers
        


class CNNwBN(torch.nn)