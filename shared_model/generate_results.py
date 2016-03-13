import os
from pos_eval import pos_eval

def generate_results():
    print('generating latex tables - chunk train')
    cmd = 'perl eval.pl -l < ../../data/current_outcome/chunk_pred_train.txt'
    os.system(cmd)

    print('generating latex tables - chunk valid')
    cmd = 'perl eval.pl -l < ../../data/current_outcome/chunk_pred_val.txt'
    os.system(cmd)

    print('generating latex tables - chunk combined')
    cmd = 'perl eval.pl -l < ../../data/current_outcome/chunk_pred_combined.txt'
    os.system(cmd)

    print('generating latex tables - chunk test')
    cmd = 'perl eval.pl -l < ../../data/current_outcome/chunk_pred_test.txt'
    os.system(cmd)


    print('generating accuracy - pos train')
    print(pos_eval('../../data/current_outcome/pos_pred_train.txt'))

    print('generating accruacy - pos valid')
    print(pos_eval('../../data/current_outcome/pos_pred_val.txt'))

    print('generating accruacy - pos combined')
    print(pos_eval('../../data/current_outcome/pos_pred_combined.txt'))

    print('generating accruacy - pos test')
    print(pos_eval('../../data/current_outcome/pos_pred_test.txt'))

    print('done')
