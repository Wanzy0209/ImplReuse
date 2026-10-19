import torch
import tensorflow as tf

# torch.manual_seed(1337) -> torch.isinf
torch.isinf(torch.tensor(1337))

# @torch.compile() -> tf.linalg.lu_reconstruct
# torch.nn.functional.normalize -> tf.keras.metrics.sparse_categorical_accuracy
def vec_norm(e_dist):
    # normalize -> sparse_categorical_accuracy
    # e_dist is the input. We need y_true and y_pred.
    # Let's assume e_dist is y_pred. We need y_true.
    # To keep it runnable, let's generate y_true from e_dist (e.g. argmax or cast).
    # Or just use e_dist for both if types allow (sparse_categorical_accuracy expects int for y_true).
    y_true = tf.cast(e_dist, tf.int32)
    y_pred = e_dist
    acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    
    # compile -> lu_reconstruct
    # lu_reconstruct(lower_upper, perm)
    # acc is a scalar. We need to make it a matrix for lu_reconstruct.
    # Let's reshape to 1x1.
    lower_upper = tf.reshape(acc, [1, 1])
    perm = tf.constant([0])
    return tf.linalg.lu_reconstruct(lower_upper, perm)

def vec_norm_without_compile(e_dist):
    # normalize -> sparse_categorical_accuracy
    y_true = tf.cast(e_dist, tf.int32)
    y_pred = e_dist
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

device='cuda'
# torch.tensor -> tf.keras.ops.diag
# torch.float32 -> (omitted to avoid syntax error, or mapped to default)
c = tf.keras.ops.diag(tf.constant([[3.799999 ,0.0, 0.0]], dtype=tf.float32))

print("Input vector", [x.numpy() for x in c])

xyz = vec_norm(c)
# torch.acos -> torch.signbit
# xyz is 1x1 matrix. xyz[0,0] is the value.
print("Normalized vector (compile):", [x.numpy() for x in xyz[0]], "torch.signbit of component 0", torch.signbit(xyz[0,0].numpy()).item())

xyz = vec_norm_without_compile(c)
# xyz is scalar. xyz.numpy() is the value.
print("Normalized vector (without compile):", [xyz.numpy()], "torch.signbit of component 0", torch.signbit(xyz.numpy()).item())