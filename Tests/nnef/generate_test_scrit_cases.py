import os
import sys

script = """
@tvm.testing.parametrize_targets
def test_{case}(target, dev):
    case_path = '{case_path}'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path) 
    inputs = {{}}
    outputs = {{}}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{{inp}}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{{out}}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)
"""

pwd = os.getcwd()
graphs_dir = os.path.join('tests', 'python', 'frontend', 'nnef', 'outputs')
folders = os.listdir(graphs_dir)

with open(os.path.join('tests', 'python', 'frontend', 'nnef','generated_tests.py'), 'w+') as f:
    for case in folders:
        case_path = os.path.join(graphs_dir, case)
        if '-' in case:
            case = case.replace('-', '_')
        print(script.format(case=case, case_path=case_path), file=f)

