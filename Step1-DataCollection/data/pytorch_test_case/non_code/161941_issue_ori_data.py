#!/usr/bin/env python3
import torch
import torch.nn as nn
import warnings
import tempfile
import os

# CRITICAL: Force DeprecationWarnings to be errors (this triggers the issue!)
warnings.filterwarnings("error", category=DeprecationWarning)

class TestModel(nn.Module):
    def forward(self, x):
        # This reshape triggers the boolean->int issue
        return x.reshape(1, -1)

model = TestModel().eval()
x = torch.randn(2, 4)

# Try ONNX export - this should trigger SerdeError
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    try:
        torch.onnx.export(
            model,
            (x,),
            f.name,
            opset_version=18,
            dynamo=True,
            verbose=False
        )
        print("❌ UNEXPECTED: Export succeeded")
    except Exception as e:
        print(f"✅ CONFIRMED: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(f.name)
        except:
            pass