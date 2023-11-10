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

- Model Parallelism: Best for models where there are independent parts of computation that you can run in parallel. 




## Synchronous and Asynchronous Data Parallelism 