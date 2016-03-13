import os
from generate_results import generate_results
print('running model')
cmd = 'python model.py'
os.system(cmd)
generate_results()
