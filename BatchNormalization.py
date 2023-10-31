import torch
from torchvision import datasets, transforms

'''
Batch Normalization on MNIST
Normalizes the inputs to a layer which can help
improve convergence speed and potentially model accuracy 
'''
# transform to normalize the data
transform = transforms.Compose([
    # Converts a PIL image or numpy array of shape H x W x C in the range [0,255]
    # to a PyTorch tensor of shape C X H X W in the range [0.0,1.0]
    # converts into PyTorch's tensor format
    transforms.ToTensor(),
    # rescales the image's pixel values 
    # transforms.Normalize(mean, std)
    # This transformation will subtract the mean and divide the std
    # grayscale img (MNIST) --> one channel, so a single val
    # but for RGB imgs, need to specify values for all three 
    transforms.Normalize((0.5,),(0.5,))
])

# Download the MNIST dataset
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

class NetWithoutBN(torch.nn.Module):
    def __init__(self):
        # three linear layers
        super(NetWithoutBN, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)


class NetWithBN(torch.nn.Module):
    def __init__(self):
        super(NetWithBN, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, 10)