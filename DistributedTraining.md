# What is Distributed Training?

- Using the right hardware configuration can dramatically reduce training time. 
- A shorter training time makes for faster iteration to reach your modeling goals. 
- Tensorflow needs to know how to process across the multiple GPUs in your runtime. 

## Data Parallelism vs. Model Parallelism 

- Data Parallelism: Works with just about any model architecture -- widely adopted for dist. training. 

Think about the batch size to understand data parallelism! 

```
model.fit(x, y, batch_size=32)
```

On each step of the model training, a batch of data is used to calculate gradients, which are then used to update the model's weights.  

Generally, the larger the batch size, the more accurate the gradients are. 

With data parallelism, we can add an additional GPU, which would allow us to double our batch size.

```
model.fit(x, y, batch_size=(32 * NUM_GPUS))
# So with 2 GPUs, your batch size is 64. With 4, 128. 
```

Each GPU gets a separate slice of the data, they calculate the gradients, and those gradients are averaged. 

Core idea of data parallelism: More GPUs, your model is able to see more data one each training step. This means less time to finish an epoch -- a full pass through the training data. 

- Model Parallelism: Best for models where there are independent parts of computation that you can run in parallel. 



## Synchronous and Asynchronous Data Parallelism 