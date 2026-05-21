import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import unittest

class TestMPSCompileBackward(unittest.TestCase):
    """
    Test case for Issue 161905: torch.compile ResNet-18 model fails during loss.backward() on MPS backend.
    
    This test leverages the code pattern from the similar API 'tf.compat.v1.summary.all_v2_summary_ops'.
    The TF API checks 'context.executing_eagerly()' to determine whether to return None or a collection of ops.
    Here, we translate this pattern to PyTorch by checking 'torch.compiler.is_compiling()' to verify
    the execution context and ensure the backward pass (graph ops) executes correctly in compiled mode on MPS.
    """

    def test_resnet18_mps_compile_backward(self):
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend not available")

        BATCH_SIZE = 4
        NUM_CLASSES = 10
        LEARNING_RATE = 0.01
        device = 'mps'

        model = resnet18(num_classes=NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        model = model.to(device)
        model.train()

        # Helper function mimicking the logic of tf.compat.v1.summary.all_v2_summary_ops
        # which checks context.executing_eagerly() to determine behavior.
        def get_execution_context_ops():
            # TF equivalent: if context.executing_eagerly(): return None
            if not torch.compiler.is_compiling():
                return None
            
            # TF equivalent: return ops.get_collection(...)
            # In PyTorch, we verify we are in the compiled context where the graph ops exist.
            return "compiled_graph_ops"

        @torch.compile
        def train_step(images, labels):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Check execution context, similar to the TF API pattern
            context_status = get_execution_context_ops()

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # The bug occurs here: backward pass in compiled mode on MPS
            loss.backward()
            
            optimizer.step()
            
            return loss, context_status

        images = torch.randn(BATCH_SIZE, 3, 224, 224)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        # Execute the compiled step
        loss, status = train_step(images, labels)
        
        # Assertions
        self.assertIsNotNone(loss, "Loss should not be None")
        
        # Verify we are in the compiled context (mimicking the TF API's logic branch)
        self.assertEqual(status, "compiled_graph_ops", 
                         "Expected execution in compiled graph mode")
        
        # Verify gradients were computed (checking the result of the backward ops)
        grad_found = False
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for parameter is None")
                grad_found = True
        
        self.assertTrue(grad_found, "No gradients were computed during backward pass")

if __name__ == '__main__':
    unittest.main()