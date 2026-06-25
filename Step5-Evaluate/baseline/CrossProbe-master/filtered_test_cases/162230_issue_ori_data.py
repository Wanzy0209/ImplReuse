import subprocess
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write('print("test")')
    f.flush()
    try:
        subprocess.run(['python', f.name], timeout=30)
    except subprocess.TimeoutExpired:
        print('Timeout occurred')