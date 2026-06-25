import sys
if sys.version_info < (3, 8):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points