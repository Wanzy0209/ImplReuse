import importlib
import sys

# Reproduce the import errors
try:
    importlib.import_module('torch._export.db.examples')
except RuntimeError as e:
    print(f'Export error: {e}')

try:
    importlib.import_module('torch.testing._internal.hop_db')
except ImportError as e:
    print(f'Import error: {e}')