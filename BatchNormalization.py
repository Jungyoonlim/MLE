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
    transforms.Normalize((0,5,),(0,5,))
])

# Load the MNIST dataset


