import sys
import os


ops = {
    'add': 'scalar',
    'sub': 'scalar',
    'mul': 'scalar',
    'div': 'scalar',
    'pow': 'scalar',
    'min': 'scalar',
    'max': 'scalar',
    'lt': 'scalar',
    'gt': 'scalar',
    'le': 'scalar',
    'ge': 'scalar',
    'eq': 'scalar',
    'ne': 'scalar',
    'and': 'logical',
    'or': 'logical',
}


template_2d = """version 1.0;

graph G( input1, input2 ) -> ( output )
{{
    input1 = external<{type}>(shape = [4,16]);
    input2 = external<{type}>(shape = [4,16]);
    output = {name}(input1, input2);
}}
"""

template_4d = """version 1.0;

graph G( input1, input2 ) -> ( output )
{{
    input1 = external<{type}>(shape = [4,16,32,32]);
    input2 = external<{type}>(shape = [4,16,32,32]);
    output = {name}(input1, input2);
}}
"""

template_4d_broadcast = """version 1.0;

graph G( input1, input2 ) -> ( output )
{{
    input1 = external<{type}>(shape = [4,16,32,32]);
    input2 = external<{type}>(shape = [1,16,1,1]);
    output = {name}(input1, input2);
}}
"""

template_4d_constant = """version 1.0;

graph G( input ) -> ( output )
{{
    input = external<{type}>(shape = [4,16,32,32]);
    output = {name}(input, {const});
}}
"""


path = sys.argv[1] if len(sys.argv) > 1 else ''


for op, type in ops.items():
    with open(os.path.join(path, '{}_2d.nnef'.format(op)), 'w') as file:
        file.write(template_2d.format(name=op, type=type))
    with open(os.path.join(path, '{}_4d.nnef'.format(op)), 'w') as file:
        file.write(template_4d.format(name=op, type=type))
    with open(os.path.join(path, '{}_4d_broadcast.nnef'.format(op)), 'w') as file:
        file.write(template_4d_broadcast.format(name=op, type=type))
    with open(os.path.join(path, '{}_4d_constant.nnef'.format(op)), 'w') as file:
        file.write(template_4d_constant.format(name=op, type=type, const=0.5 if type == 'scalar' else 'false'))
