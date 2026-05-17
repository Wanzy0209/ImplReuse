import torch
import tensorflow as tf
from transformers import AutoTokenizer

def test_tf_keras_input_with_tokenizer():
    """
    Adapts the PyTorch export test case to use tf.keras.Input.
    Instead of exporting a pre-existing model with concrete inputs,
    we define the model's input layers symbolically based on the 
    tokenizer's output shape.
    """
    # Setup tokenizer (same as original PyTorch test case)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    # Prepare inputs to get the shape requirements
    # Using return_tensors="np" to easily extract shapes for the TF Input definition
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )

    # Extract shapes from the tokenized inputs
    # inputs["input_ids"] shape is typically (batch_size, sequence_length)
    batch_size = inputs["input_ids"].shape[0]
    sequence_length = inputs["input_ids"].shape[1]

    # Adaptation: Use tf.keras.Input to define the input signature
    # This is the TensorFlow equivalent of specifying the input constraints 
    # that torch.export.export would infer from example_inputs.
    
    input_ids = tf.keras.Input(
        shape=(sequence_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="input_ids"
    )

    attention_mask = tf.keras.Input(
        shape=(sequence_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="attention_mask"
    )

    # Construct a minimal model to verify the inputs are valid
    # In the original bug, torch.export.export failed to handle the mode.
    # Here, we verify that tf.keras.Input successfully creates the tensor 
    # specifications for the model.
    
    # Simple embedding layer to consume input_ids
    x = tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128)(input_ids)
    
    # We pass attention_mask through or use it in a layer that supports masking
    # For this minimal test, we just ensure it is part of the Model's inputs
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask], 
        outputs=outputs
    )

    # Assertions to verify behavior
    assert model is not None, "Model construction failed with tf.keras.Input"
    assert len(model.inputs) == 2, "Model should have two inputs (input_ids, attention_mask)"
    
    # Verify input shapes match expectations
    assert model.inputs[0].shape == (batch_size, sequence_length)
    assert model.inputs[1].shape == (batch_size, sequence_length)
    
    print("Test passed: tf.keras.Input successfully defined model structure.")

if __name__ == "__main__":
    test_tf_keras_input_with_tokenizer()