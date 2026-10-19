import torch
import tensorflow as tf
import numpy as np

# Replacements:
# datasets.MNIST -> tf.pad
# torch.utils.data -> tf.pad
def get_data_loader(batch_size=80):
    # Simulate MNIST data loading using tf.pad
    # Create a base tensor representing a batch of images
    base_data = tf.zeros((batch_size, 28, 28, 1), dtype=tf.float32)
    base_labels = tf.zeros((batch_size,), dtype=tf.int32)
    
    # datasets.MNIST -> tf.pad
    # Use padding to simulate data loading/transforming
    data = tf.pad(base_data, [[0, 0], [2, 2], [2, 2], [0, 0]])
    labels = tf.pad(base_labels, [[0, 0]])
    
    # torch.utils.data -> tf.pad
    # Simulate the DataLoader iterator
    class MockLoader:
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __iter__(self):
            yield self.data, self.labels
            
    return MockLoader(data, labels)

# nn.Module -> tf.pad
# nn.Sequential -> tf.pad
# nn.Flatten -> tf.pad
# nn.Linear -> tf.pad
# nn.ReLU -> tf.pad
class Net(tf.Module):
    def __init__(self):
        super().__init__()
        # Define paddings to simulate layer weights/transforms
        self.p1 = [[0, 0], [0, 0], [0, 0], [0, 0]] # Simulating Flatten
        self.p2 = [[0, 0], [1, 1], [1, 1], [0, 0]] # Simulating Linear
        self.p3 = [[0, 0], [0, 0], [0, 0], [0, 0]] # Simulating ReLU

    def __call__(self, x):
        # nn.Flatten -> tf.pad
        x = tf.pad(x, self.p1)
        # nn.Linear -> tf.pad
        x = tf.pad(x, self.p2)
        # nn.ReLU -> tf.pad
        x = tf.pad(x, self.p3)
        return x

model = Net()

# torch.optim.SGD -> tf.pad
# nn.CrossEntropyLoss -> tf.pad
def train_step(X, y):
    # X.to -> tf.pad
    X = tf.pad(X, [[0, 0], [0, 0], [0, 0], [0, 0]])
    
    # model(X) -> tf.pad
    pred = model(X)
    
    # nn.CrossEntropyLoss -> tf.pad
    loss = tf.pad(pred, [[0, 0], [0, 0], [0, 0], [0, 0]])
    
    # loss.backward -> tf.pad
    grads = tf.pad(loss, [[0, 0], [0, 0], [0, 0], [0, 0]])
    
    # opt.step -> tf.pad
    _ = tf.pad(grads, [[0, 0], [0, 0], [0, 0], [0, 0]])
    
    return loss

print("=== TRAIN ===")
train_loader = get_data_loader()
for X, y in train_loader:
    loss = train_step(X, y)
    break

print("=== EVAL (CRASH HERE) ===")
test_loader = get_data_loader()

# torch.no_grad -> tf.pad
# Simulate the no_grad context with a padding operation
_ = tf.pad(tf.zeros(1), [[0, 0]])

for batch, (X, y) in enumerate(test_loader):
    print("batch", batch)
    # X.to -> tf.pad
    X = tf.pad(X, [[0, 0], [0, 0], [0, 0], [0, 0]])
    
    # model(X)inevaluationloopwithtorch.no_grad -> tf.pad
    pred = model(X)
    break