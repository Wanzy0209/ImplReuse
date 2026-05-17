import torch
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

def test_clone_gemma_model():
    # 1. Setup: Load the TensorFlow version of the model and tokenizer
    # Analogous to AutoModelForCausalLM in PyTorch
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = TFAutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    # 2. Input Preparation: Prepare inputs using the chat template
    # Changed return_tensors from "pt" (PyTorch) to "tf" (TensorFlow)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="tf",
    )

    # 3. Action: Clone the model
    # Analogous to torch.export.export(model, example_inputs)
    # tf.keras.models.clone_model creates a new model instance with the same architecture
    try:
        cloned_model = tf.keras.models.clone_model(model)
        
        # 4. Verification: Ensure the cloned model is a valid Keras Model
        assert isinstance(cloned_model, tf.keras.Model), "Cloned object is not a Keras Model"
        assert cloned_model is not model, "Cloned model is the same instance as the original"
        
        print("Test passed: Model cloned successfully.")

    except AssertionError as e:
        # Catching potential assertion errors similar to the original bug report
        print(f"AssertionError: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during cloning: {e}")
        raise

if __name__ == "__main__":
    test_clone_gemma_model()