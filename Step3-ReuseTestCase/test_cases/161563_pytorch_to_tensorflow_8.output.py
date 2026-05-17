import torch
import tensorflow as tf
import tempfile
import shutil
import os

def test_rnn_export():
    """
    Adapted test case for tf.keras.layers.RNN based on the PyTorch export bug.
    Original Bug: AssertionError when exporting a model with torch.export.export.
    Target API: tf.keras.layers.RNN
    Semantic Translation: 
        - PyTorch: Load model -> Prepare inputs -> torch.export.export(model, inputs)
        - TensorFlow: Define model with RNN -> Prepare inputs -> tf.saved_model.save (Export)
    """
    
    # 1. Define a model using the target API: tf.keras.layers.RNN
    # We wrap an LSTM cell in the RNN layer to create a recurrent layer instance.
    cell = tf.keras.layers.LSTMCell(64)
    rnn_layer = tf.keras.layers.RNN(cell, return_sequences=False, return_state=False)

    # Build a simple Keras Model
    inputs = tf.keras.Input(shape=(None, 32), batch_size=1, name="input_ids")
    outputs = rnn_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 2. Prepare example inputs
    # Mimicking the input preparation from the original bug report (batch_size=1, seq_len=10)
    batch_size = 1
    seq_length = 10
    feature_dim = 32
    example_inputs = tf.random.normal((batch_size, seq_length, feature_dim))

    # Ensure the model is built/ran once (optional but good practice for TF)
    _ = model(example_inputs)

    # 3. Attempt to export the model
    # In PyTorch, the bug occurred during torch.export.export(model, args).
    # In TensorFlow, the equivalent is saving the model via tf.saved_model.save,
    # which traces the graph and serializes it.
    export_dir = tempfile.mkdtemp()

    try:
        print(f"Attempting to export model with {rnn_layer.__class__.__name__}...")
        
        # This triggers the graph capture and export logic
        tf.saved_model.save(model, export_dir)
        
        print("Export successful.")
        
        # Verify the export worked by loading it back
        loaded_model = tf.saved_model.load(export_dir)
        infer = loaded_model.signatures["serving_default"]
        
        # Run inference on the loaded model to ensure integrity
        result = infer(input_ids=example_inputs)
        assert result is not None, "Exported model produced no output."
        
        print("Verification successful: Exported model runs correctly.")

    except AssertionError as e:
        # Catching the specific error type mentioned in the original bug report
        print(f"AssertionError during export (similar to original bug): {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during export: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

if __name__ == "__main__":
    test_rnn_export()