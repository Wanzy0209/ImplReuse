import subprocess
try:
    subprocess.run(['python', '/tmp/tmp1pgf03ma.py'], timeout=30)
except subprocess.TimeoutExpired:
    print('Test timed out after 30 seconds')