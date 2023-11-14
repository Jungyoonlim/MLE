import torch
import numpy as np

'''
x = torch.rand(5,3)
print(x)
print(x[:,0])
print(x[1,:])
print(x[1,1])
print(x[1,1].item())

x = torch.rand(4,4)
print(x)

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y 
    z = z.to("cpu")

# Later requires gradients 
x = torch.ones(5, requires_grad=True)
print(x)
'''

# Calculate Gradients 
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2 

'''
x --- ( + ) -- y
2 -----|
'''

