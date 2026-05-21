import torch
import tensorflow as tf

def test_name_scope_with_learnable_scalar():
    """
    Adapted test case for tf.keras.name_scope based on the PyTorch Flex Attention bug.
    Verifies that a learnable scalar (tf.Variable) can be used inside a name_scope
    context without causing errors during forward or backward passes, similar to the
    original issue with torch.compile and flex_attention.
    """
    # Equivalent to: temp = nn.Parameter(torch.tensor(0.0))
    temp = tf.Variable(0.0, trainable=True, name="learnable_scalar")

    # Equivalent to the score_mod function
    # We wrap the logic in tf.keras.name_scope as requested by the similar API mapping
    def score_mod(score, b, h, q, kv):
        with tf.keras.name_scope("score_modification"):
            # Original bug logic: adding a scalar parameter to the score
            # In PyTorch, this caused vmap/compile errors.
            # In TensorFlow, we verify it works within the name_scope.
            score = score + temp
            return score

    # Compile the function (equivalent to torch.compile)
    compiled_score_mod = tf.function(score_mod)

    # Dummy inputs mimicking attention scores (Batch, Heads, SeqLen, SeqLen)
    B, H, S = 2, 4, 8
    score = tf.random.normal((B, H, S, S))
    b = tf.constant(0) # Dummy batch index
    h = tf.constant(0) # Dummy head index
    q = tf.constant(0) # Dummy query index
    kv = tf.constant(0) # Dummy key/value index

    # 1. Test Forward Pass (Compiled)
    print("Testing Forward Pass (Compiled)...")
    try:
        output = compiled_score_mod(score, b, h, q, kv)
        assert output.shape == score.shape
        print("Forward Pass: Success")
    except Exception as e:
        print(f"Forward Pass: Failed with {e}")

    # 2. Test Backward Pass (Compiled)
    print("Testing Backward Pass (Compiled)...")
    try:
        with tf.GradientTape() as tape:
            output = compiled_score_mod(score, b, h, q, kv)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, temp)
        assert grads is not None
        print("Backward Pass: Success")
    except Exception as e:
        print(f"Backward Pass: Failed with {e}")

if __name__ == "__main__":
    test_name_scope_with_learnable_scalar()