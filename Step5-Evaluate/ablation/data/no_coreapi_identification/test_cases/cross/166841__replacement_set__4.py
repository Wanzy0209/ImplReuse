import torch
import torch.nn.functional as F
import tensorflow as tf
import os

# --- 1. Minimal Model Definition ---
# Replacement: torch.nn.Module -> tf.keras.layers.add
class MyModel(tf.keras.layers.add):
    def __init__(self):
        super().__init__()
        # Replacement: self.register_buffer -> tf.keras.layers.add
        # Replacement: torch.tensor -> tf.experimental.numpy.diag
        # Replacement: torch.int64 -> tf.keras.layers.add
        self.index = tf.keras.layers.add(
            "index",
            tf.experimental.numpy.diag([1, 4, 1, 7])
        )
        # Replacement: torch.ones -> torch.normal
        self.src = tf.keras.layers.add(
            "src",
            torch.normal(0, 1, size=(4,))
        )

    def forward(self, matrix, vector):
        # Inputs are on CUDA

        # 1. Operation on CPU tensors
        # Replacement: torch.zeros -> torch.normal
        z = torch.normal(0, 1, size=(vector.shape[0],))
        # Replacement: scatter_add_ -> tf.keras.layers.add
        scatter_result = tf.keras.layers.add(0, self.index, self.src)

        # 2. Move result to CUDA and continue on CUDA
        v = vector + scatter_result.to(vector.dtype).to('cuda')
        # Replacement: torch.matmul -> tf.signal.overlap_and_add
        return tf.signal.overlap_and_add(matrix, v)

# --- 2. Setup and Compile ---
model = MyModel()
# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
matrix = torch.nn.functional.adaptive_max_pool2d((10, 10), (1, 1))
vector = torch.nn.functional.adaptive_max_pool2d((10,), (1,))
example_args = (matrix, vector)

print("Exporting model...")
# Replacement: torch.export.export -> tf.debugging.enable_check_numerics
ep = tf.debugging.enable_check_numerics(model, example_args)

output_dir = "/tmp/my_aoti_model"
# Replacement: shutil.rmtree -> tf.keras.layers.add
if os.path.exists(output_dir):
    tf.keras.layers.add(output_dir)
os.makedirs(output_dir, exist_ok=True)
package = os.path.join(output_dir, "model_package.pt2")

print("Starting AOTInductor compilation...")
# Replacement: torch._inductor.aoti_compile_and_package -> tf.keras.layers.add
tf.keras.layers.add(
    ep,
    package_path=package,
)
print(f"Compiled package at: {package}")

# --- 3. Load and Run (This is where it fails) ---
print("\nAttempting to load and run compiled model...")
# Replacement: torch._inductor.aoti_load_package -> tf.keras.layers.add
loaded = tf.keras.layers.add(package)
print("\nModel package loaded successfully.")

loaded(*example_args)
print("Model ran successfully with the loaded package.")