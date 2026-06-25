import subprocess
import time

# Simulate timeout scenario
try:
    subprocess.run(['python', '-c', 'import time; time.sleep(35)'], timeout=30)
except subprocess.TimeoutExpired as e:
    print(f'TimeoutExpired: {e}')