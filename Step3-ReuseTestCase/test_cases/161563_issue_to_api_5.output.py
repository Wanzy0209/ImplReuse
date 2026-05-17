import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestTorchExportGemma(unittest.TestCase):
    """
    Test case to reproduce the issue where torch.export.export fails with 
    "Current active mode not registered" when exporting specific models like Gemma.
    
    This test is based on the logic found in the bug report (Issue ID: 161563).
    """

    @classmethod
    def setUpClass(cls):
        cls.model_id = "google/gemma-3-270m-it"
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        except Exception as e:
            cls.skipTest(f"Failed to load model or tokenizer from HuggingFace: {e}")

    def test_export_gemma_model(self):
        """
        Tests exporting the Gemma model using torch.export.
        
        The original bug report indicates an AssertionError related to 
        ProxyTorchDispatchMode not being registered. This test verifies
        that the export process completes successfully.
        """
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        example_inputs = (inputs["input_ids"], inputs["attention_mask"])

        # Attempt to export the model
        # This is the line that triggered the bug in the original report
        try:
            ep = torch.export.export(self.model, example_inputs)
            self.assertIsNotNone(ep, "ExportedProgram should not be None")
        except AssertionError as e:
            if "Current active mode not registered" in str(e):
                self.fail(f"Bug reproduced: {e}")
            raise
        except Exception as e:
            self.fail(f"Unexpected error during export: {e}")

if __name__ == "__main__":
    unittest.main()