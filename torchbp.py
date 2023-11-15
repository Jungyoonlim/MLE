import torch

# https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4

x = torch.tensor(1,0)
y = torch.tensor(2,0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute loss
y_hat = w * x 
loss = (y_hat - y)**2

print(loss)

# backward pass 
loss.backward()
print(w.grad)