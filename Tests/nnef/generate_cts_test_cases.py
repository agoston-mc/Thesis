import os
import sys

'''
usage: 
    go to tvm/tests/python/frontend/nnef, and run as 
    `python generate_test_scrit_cases.py`
    it will generate generated_tests.py with test cases in current wd
        from ./outputs with default tolerances of 1e-5

'''

script = """
@tvm.testing.parametrize_targets
def test_cts_{case}(target, dev):
    verify_model('{case_path}', target, dev, rtol=1e-5, atol=1e-5)
"""

pwd = os.getcwd()
graphs_dir = os.path.join(pwd, 'outputs')
folders = os.listdir(graphs_dir)

with open(os.path.join(pwd, 'generated_tests.py'), 'w+') as f:
    for case in folders:
        case_path = case
        if '-' in case:
            case = case.replace('-', '_')
        print(script.format(case=case, case_path=case_path), file=f)

