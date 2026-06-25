import subprocess
import tempfile
import os

# Create a temporary Python script that might hang
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write('''
import time
# Simulate work that might take >30 seconds
time.sleep(35)
print("Done")
''')
    temp_script = f.name

try:
    # This will timeout after 30 seconds
    result = subprocess.run(
        ['python', temp_script],
        capture_output=True,
        text=True,
        timeout=30
    )
except subprocess.TimeoutExpired as e:
    print(f"Timeout after 30 seconds: {e}")
finally:
    os.unlink(temp_script)