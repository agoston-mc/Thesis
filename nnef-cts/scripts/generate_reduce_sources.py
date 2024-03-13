import sys
import os


ops = {
    'min_reduce': 'scalar',
    'max_reduce': 'scalar',
    'sum_reduce': 'scalar',
    'mean_reduce': 'scalar',
    'any_reduce': 'logical',
    'all_reduce': 'logical',
    'argmin_reduce': 'scalar',
    'argmax_reduce': 'scalar',
}


template_channel = """version 1.0;

graph G( input ) -> ( output )
{{
    input = external<{type}>(shape = [4,16,32,32]);
    output = {name}(input, axes = [1]);
}}
"""

template_spatial = """version 1.0;

graph G( input ) -> ( output )
{{
    input = external<{type}>(shape = [4,16,32,32]);
    output = {name}(input, axes = [2,3]);
}}
"""


path = sys.argv[1] if len(sys.argv) > 1 else ''


for op, type in ops.items():
    with open(os.path.join(path, '{}_channel.nnef'.format(op)), 'w') as file:
        file.write(template_channel.format(name=op, type=type))
    if not op.startswith('arg'):
        with open(os.path.join(path, '{}_spatial.nnef'.format(op)), 'w') as file:
            file.write(template_spatial.format(name=op, type=type))
