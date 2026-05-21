import torch
import tensorflow as tf
import os

def test_tf_lite_analyzer_with_squeeze_logic():
    """
    Adapts the PyTorch reproduction logic (squeeze -> stack -> reshape) 
    to TensorFlow and verifies it using tf.lite.experimental.Analyzer.
    """
    
    # 1. Define the TensorFlow function equivalent to the PyTorch fuzzed_program
    # PyTorch: chunk -> squeeze -> stack -> reshape
    @tf.function
    def tf_model(arg_0):
        # var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0]
        # TF equivalent: split tensor into 4 parts along axis 0 and take the first
        splits = tf.split(arg_0, 4, axis=0)
        var_node_3 = splits[0]  # Shape: (1,)

        # var_node_2 = torch.squeeze(var_node_3)
        # TF equivalent: squeeze removes dimensions of size 1
        var_node_2 = tf.squeeze(var_node_3)  # Shape: ()

        # var_node_1 = torch.stack([var_node_2], dim=0)
        # TF equivalent: stack tensors along a new axis
        var_node_1 = tf.stack([var_node_2], axis=0)  # Shape: (1,)

        # var_node_0 = torch.reshape(var_node_1, [1])
        # TF equivalent: reshape tensor
        var_node_0 = tf.reshape(var_node_1, [1])  # Shape: (1,)

        return var_node_0

    # 2. Setup inputs
    # PyTorch: arg_0 = torch.as_strided(..., (4,), (1,)) -> bool tensor of size 4
    # TF: Create a boolean tensor of shape (4,)
    arg_0 = tf.constant([True, False, True, False], dtype=tf.bool)

    # 3. Run Eager (corresponds to PyTorch eager execution)
    try:
        result_eager = tf_model(arg_0)
        print(' eager success')
        print(f"   Result shape: {result_eager.shape}, dtype: {result_eager.dtype}")
    except Exception as e:
        print(f' eager failed: {e}')
        return

    # 4. Compile (Convert to TFLite)
    # This corresponds to torch.compile in the original issue
    tflite_model_path = 'temp_model.tflite'
    try:
        # Get the concrete function to trace the graph
        concrete_func = tf_model.get_concrete_function(arg_0)
        
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func], 
            tf_model
        )
        # Enable standard ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        print(' compile (tflite conversion) success')

        # Save the model to a temporary file for the Analyzer
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        # 5. Run the Similar API: tf.lite.experimental.Analyzer
        # This corresponds to verifying the compiled artifact's structure
        try:
            # The Analyzer inspects the flatbuffer model for issues
            analysis_report = tf.lite.experimental.Analyzer.analyze(model_path=tflite_model_path)
            print(' analyzer success')
            
            # Print a snippet of the analysis to verify behavior
            # If there were internal stride/shape issues like in PyTorch, they might appear here
            print("Analysis Report (first 500 chars):")
            print(analysis_report[:500])
            
        except Exception as e:
            print(f' analyzer failed: {e}')
            
    except Exception as e:
        print(f' compile (tflite conversion) failed: {e}')
    finally:
        # Cleanup
        if os.path.exists(tflite_model_path):
            os.remove(tflite_model_path)

if __name__ == '__main__':
    test_tf_lite_analyzer_with_squeeze_logic()