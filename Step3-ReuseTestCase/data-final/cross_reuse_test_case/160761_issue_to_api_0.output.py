import unittest
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestGemma3ONNXExport(unittest.TestCase):
    """
    Test case for Issue 160761: Gemma3 ONNX Export Fails due to vmap.
    This test leverages the pattern from torch.backends.cudnn.version for environment checking.
    """

    def _check_environment(self):
        """
        Adapted pattern from torch.backends.cudnn.version:
        Check initialization/state before proceeding.
        """
        # Leverage the similar API: torch.backends.cudnn.version
        cudnn_version = torch.backends.cudnn.version()
        
        # Log environment info
        print(f"PyTorch Version: {torch.__version__}")
        print(f"cuDNN Version: {cudnn_version}")
        
        # Check if we can access the model (similar to _init check)
        try:
            AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
            return True
        except Exception as e:
            print(f"Environment check failed: {e}")
            return False

    def test_onnx_export_with_dynamo(self):
        # Apply the pattern: if not check, return/skip
        if not self._check_environment():
            self.skipTest("Environment check failed or model not accessible.")

        model_id = "google/gemma-3-270m-it"
        
        # Load Model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Prepare Inputs
        messages = [{"role": "user", "content": "Who are you?"}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        example_inputs = (inputs["input_ids"], inputs["attention_mask"])
        output_path = "gemma3.onnx"

        # Attempt Export
        # This is expected to trigger the vmap issue in transformers==4.55.0
        try:
            torch.onnx.export(
                model,
                example_inputs,
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=True,
            )
            # If successful, verify file creation
            self.assertTrue(os.path.exists(output_path))
        except Exception as e:
            # For bug reproduction, we might expect this to fail.
            # We print the error to confirm the bug behavior.
            print(f"Export failed as expected in buggy version: {e}")
            # Depending on the test goal (reproduction vs regression), 
            # we might re-raise or assert. Here we log to preserve logic.
            raise
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    unittest.main()