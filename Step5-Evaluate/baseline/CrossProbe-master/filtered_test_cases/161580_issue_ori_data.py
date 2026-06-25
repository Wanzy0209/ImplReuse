import subprocess
try:
    subprocess.run(['python', 'test_script.py'], timeout=30)
except subprocess.TimeoutExpired:
    print('Test timed out')