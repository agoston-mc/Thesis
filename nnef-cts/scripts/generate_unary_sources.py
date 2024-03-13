import sys
import os


ops = {
    'copy': 'scalar',
    'neg': 'scalar',
    'rcp': 'scalar',
    'exp': 'scalar',
    'log': 'scalar',
    'sin': 'scalar',
    'cos': 'scalar',
    'tan': 'scalar',
    'asin': 'scalar',
    'acos': 'scalar',
    'atan': 'scalar',
    'sinh': 'scalar',
    'cosh': 'scalar',
    'tanh': 'scalar',
    'asinh': 'scalar',
    'acosh': 'scalar',
    'atanh': 'scalar',
    'abs': 'scalar',
    'sign': 'scalar',
    'floor': 'scalar',
    'ceil': 'scalar',
    'round': 'scalar',
    'not': 'logical',
    'sqr': 'scalar',
    'sqrt': 'scalar',
    'rsqr': 'scalar',
    'rsqrt': 'scalar',
    'log2': 'scalar',
}


template_2d = """version 1.0;

graph G( input ) -> ( output )
{{
    input = external<{type}>(shape = [4,16]);
    output = {name}(input);
}}
"""

template_4d = """version 1.0;

graph G( input ) -> ( output )
{{
    input = external<{type}>(shape = [4,16,32,32]);
    output = {name}(input);
}}
"""


path = sys.argv[1] if len(sys.argv) > 1 else ''


for op, type in ops.items():
    with open(os.path.join(path, '{}_2d.nnef'.format(op)), 'w') as file:
        file.write(template_2d.format(name=op, type=type))
    with open(os.path.join(path, '{}_4d.nnef'.format(op)), 'w') as file:
        file.write(template_4d.format(name=op, type=type))
