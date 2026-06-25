import os
try:
    os.listdir('/proc/2/map_files')
except FileNotFoundError:
    print('proc filesystem not accessible')