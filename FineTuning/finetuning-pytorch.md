# Fine-tuning in PyTorch 

https://rumn.medium.com/part-1-ultimate-guide-to-fine-tuning-in-pytorch-pre-trained-model-and-its-configuration-8990194b71e

Defining a model takes
- select the appropriate architecture
- customize the model head
- configure the loss function and learning rate
- setting the desired flaoting point precision 
- determining which layers to freeze or fine-tune 

## 1. Loading a pre-trained model
    1. Have a clear understanding of your specific problem and choose an appropriate architecture. 
        1. e.g. fine-tuning for classification and low latency is a priority —> MobileNet would be a good choice.
        2. Key modification required — to adjust the fully connected FC layer 
## 2. Modifying model head 
    1. modify the model’s head to adapt to the new task and utilize the valuable features it has learned 
## 3. Setting optimizer, learning rate, weight decay, and momentum 
    1. learning rate, loss function, and optimizer are interrelated components that collectively influence the model’s ability to adapt to the new task while leveraging the knowledge acquired from pre-training 
    2. Optimizers — https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e determines the algorithm to be used to update the model’s parameters based on the gradients computed during backprop. e.g. SGD, Adam, RMSprop … different parameters update rules and convergence properties. 
        1. Weight decay (L2 regularization) — prevent overfitting and encourage the model to learn simpler and momentum (Accelerate convergence and escape local minima) 
    3. Learning Rate — A hyperparameter that determines the step size at each iteration during optimization. Controls how much the model’s parameters are updated in response to the calculated gradients during backprop. Setting it too high can cause the optimization process to oscillate or diverge while setting it too low can cause slow convergence or getting trapped in local minima. 
## 4. Choosing loss functions
    1. Measures the difference or gap between the model’s predicted outputs and the actual correct answers. 
    2. A way to understand how well the model is performing on the task. 
    3. e.g. for classification tasks, cross-entropy loss is commonly used; regression — mean squared error. 
    4. Right loss fn ensures that the model focuses on optimizing the desired objective during training. 
    5. Need to consider:
        1. Custom loss function — modify or customize the loss function to suit specific requirements
        2. Metric-based loss — design or adapt the loss function to directly optimize for these metrics 
        3. Regularization — L1 and L2. L2 can be applied by setting the weight_decay value in the optimizer.

L2 

### Define a loss function
criterion = nn.CrossEntropyLoss()

### L2 regularization 
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

L1

### Define a loss function 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

### Inside the training loop 
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)

### L1 Regularization 
regularization_loss = 0.0
for param in model.parameters():
	regularization_loss += torch.norm(param, 1)
loss += 0.01 * regularization_loss

## 5. Freezing Full or Partial Network
    1. Freezing = fixing the weight of specific layer or entire network during fine tuning process. 
    2. Network freezing allows us to retain the knowledge captured by the pre-trained model while only updating certain layers to adapt to the target task.
    3. Deciding whether you should freeze all layers or partial layers of the pre-trained model before fine-tuning boils down to your specific target task. 
    4. e.g. large-scale dataset — freezing the entire entwork can help preserve the learned representations, preventing them from being overwritten. Only the model’s head is modified and trained from scratch. oth, freezing only a portion of the network can be useful when the pre-trained model’s lower layers capture general features that are likely to be relevant for the new task. 
    5. Can access individual layers or modules within the model and set their requires_grad attribute to False. This prevents the gradients from being computed and the weights from being updated during the backward pass.
e.g. Freezing entire network

for param in model.parameters():
	param.requires_grad = False

num_classes = 10 
model.fc = nn.Linear(model.fc.in_features, num_classes)

Only convolutional layers

for param in model.parameters():
	if isinstance(param, nn.Conv2d):
		param.requires_grad = False

num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes) 

### 6. Define model floating-point precision 
i) 32-bit (float32) 
- Wide dynamic range and high numerical precision 

ii) 16-bit (float16) 
- Reduce the memory footprint and computational reqs of the model 

Mixed Precision Training 

```
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

# Define your model and optimizer 
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss function 
criterion = nn.CrossEntropyLoss()

# Define scaler for automatic scaling of gradients
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
	for batch_idx, (data, targets) in enumerate(train_loader):
		data, targets = data.to(device), targets.to(device)
		
		optimizer.zero_grad()
		
		# autocasting for mixed precision 
		with autocast():
			outputs = model(data)
			loss = criterion(outputs, targets)
	
		# perform backward pass and gradient scaling
		scaler.scale(loss).backward()
	
		# update model parameters using gradients calculated during the backward pass. 
		scaler.step(optimizer)
		# adjusts the scale factor used by the scaler for the next iteration. Prevents underflow or overflow by dynamically adjusting the scale based on the gradients magnitude. 
		scaler.update()

		# print training progress
		if batch_idx % log_interval == 0:
			print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
```

### 7. Training and Validation Mode 
- Training mode
    - Enables specific operations that are required during the training process e.g. computing gradients, updating parameters, applying regularization techniques (dropout). 
- Validation mode
    - Disables certain operations that are only necessary during training e.g. computing gradients, dropout, updating parameters. 
- Very important to set model to right mode during fine-tuning as setting the model to the right mode ensures consistent behavior and proper operations for each phase. 

Single GPU and Multiple GPU 
- GPUs are essential for DL and fine-tuning tasks b/c of its high performance in parallel computations — which speed up the training process. 

```
model = MyModel()
model = model.to(device) # Moving the model to the desired device

# if multiple GPUs are available.
if torch.cuda.device_count() > 1:
	print(“Using”, torch.cuda.device_count(), “GPUs for training.”)
	model = nn.DataParallel(model) 

# Define your loss fn and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```










