import os
from generate_results import generate_results
print('running joint model')
cmd = 'python model.py --model_type "JOINT"'
os.system(cmd)
cmd = 'cp -r ../../data/current_outcome ../../data/joint/March20'
os.system(cmd)

print('running pos model')
cmd = 'python model.py --model_type "POS"'
os.system(cmd)
cmd = 'cp -r ../../data/current_outcome ../../data/pos_single/March20'
os.system(cmd)

print('running chunk model')

cmd = 'python model.py --model_type "CHUNK"'
os.system(cmd)
cmd = 'cp -r ../../data/current_outcome ../../data/chunk_single/March20'
os.system(cmd)






# Generate the results
# generate_results('../../data/current_outcome/predictions')
