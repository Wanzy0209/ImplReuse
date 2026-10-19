import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_default_model():
    return TestModel()

def get_sample_inputs():
    # Replacement: torch.device -> tf.experimental.numpy.isreal
    # Original: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We perform the replacement call. The result is not used for device placement
    # because the replacement API (cosine_embedding_loss) does not take a device argument directly.
    _ = tf.experimental.numpy.isreal(np.array([1.0]))

    # Determine device string for tensor creation
    dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Replacement: torch.randn -> torch.nn.functional.cosine_embedding_loss
    # Original: x = torch.randn(4, 10, requires_grad=True, device=device)
    # We create dummy inputs for the loss function.
    input1 = torch.ones(4, 10, device=dev_str)
    input2 = torch.ones(4, 10, device=dev_str)
    target = torch.ones(4, device=dev_str)
    x = F.cosine_embedding_loss(input1, input2, target)
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    device = inputs[0].device
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Note: The input x is now a scalar (loss result), but the model expects (4, 10).
        # This will cause a runtime error in the forward pass, but we follow the replacement logic.
        original_output = model(*inputs)
        print('Original model output shape:', original_output.shape)

        if torch.cuda.is_available():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compiled_model = torch.compile(model)
                captured_output = compiled_model(*inputs)
            
            # Note: graph.run() returns None in standard PyTorch, but we keep the original logic.
            graph_output = graph.run()
            print('Captured graph output shape:', graph_output.shape)

            # Replacement: torch.allclose -> tf.compat.v1.image.resize_nearest_neighbor
            # Original: assert torch.allclose(original_output, graph_output)
            # We attempt to construct a valid call to the replacement API.
            if graph_output is not None:
                try:
                    # Convert PyTorch tensors to TensorFlow tensors
                    t1 = tf.convert_to_tensor(original_output.cpu().numpy())
                    t2 = tf.convert_to_tensor(graph_output.cpu().numpy())
                    
                    # Stack to create a batch dimension
                    images = tf.stack([t1, t2])
                    
                    # Ensure 4D shape [batch, height, width, channels] for the image API
                    rank = len(images.shape)
                    if rank == 3:
                        images = tf.expand_dims(images, -1)
                    elif rank == 2:
                        images = tf.expand_dims(tf.expand_dims(images, 1), -1)
                    
                    # Call the replacement API
                    tf.compat.v1.image.resize_nearest_neighbor(images, size=[1, 1])
                except Exception as e:
                    print(f"Error during replacement API call: {e}")
        else:
            print("CUDA not available, skipping graph capture")

if __name__ == "__main__":
    main()