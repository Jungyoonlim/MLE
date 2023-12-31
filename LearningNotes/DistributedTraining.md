# What is Distributed Training?

- Using the right hardware configuration can dramatically reduce training time. 
- A shorter training time makes for faster iteration to reach your modeling goals. 
- Tensorflow needs to know how to process across the multiple GPUs in your runtime. 

## Data Parallelism vs. Model Parallelism 

### Data Parallelism
Works with just about any model architecture -- widely adopted for dist. training. 

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

### Model Parallelism

Best for models where there are independent parts of computation that you can run in parallel. 

Put different layers of your model on different machines or devices. 

X --> mat mul (gpu:0) --> add (gpu:1)

Model parallelism works best for models where there are independent parts of computation that you can run in parallel.  


### Combination of data and model parallelism
```
X - split -> mat mul (gpu:0) --> add (gpu:1)
  |
  |________> mat mul (gpu:2) --> add (gpu:3)

```

## Synchronous and Asynchronous Data Parallelism 

Synchronous and asynchronous refer to how the model parameters are updated. 

```
# how you can add distribution to TensorFlow code
# Machine with 2 GPUs 
BATCH_SIZE = 64

train_data = tf.data.Dataset(...).batch(BATCH_SIZE)

model = tf.keras.Sequential(...)
model.compile(...)

model.fit(train_data, epocs=5)
```
For this example,
- Dataset size: 768
- Batch size: 64
- Steps per epoch = 768/64=12

Let's update to make use of the second GPU. 

```
strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
train_data = tf.data.Dataset(...).batch(GLOBAL_BATCH_SIZE)
# Here, 32 would be the per replica batch size and 64 is a global batch size. 

with strategy.scope():
    model = tf.keras.Sequential(...)
    model.compile(...)

"""
When we call model.fit, we will make a copy, known as a replica,
of the model of both of the GPUs, and the CPU is responsible for 
preparing the tf data set batches and sending the data to GPUs. 
"""
model.fit(train_data, epocs = 5)
```

### Reducing Gradients



Notes from: 
- https://www.youtube.com/watch?v=S1tN9a4Proc&t=420s
- https://www.youtube.com/watch?v=hc0u4avAkuM
- https://www.youtube.com/watch?v=ILBPCi6Il1U


