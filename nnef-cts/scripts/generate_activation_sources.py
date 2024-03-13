import sys
import os


ops = [
    'relu',
    'elu',
    'selu',
    'gelu',
    'silu',
    'leaky_relu',
    'sigmoid',
    'tanh',
]


template_2d = """version 1.0;

graph G( input ) -> ( output )
{{
	input = external(shape = [4,16]);
	output = {name}(input);
}}
"""

template_4d = """version 1.0;

graph G( input ) -> ( output )
{{
	input = external(shape = [4,16,32,32]);
	output = {name}(input);
}}
"""

template_linear = """version 1.0;

graph G( input ) -> ( output )
{{
	input = external(shape = [4,16,32,32]);
	filter = constant(shape = [16,1,1,1], value = [1.0]);
	bias = constant(shape = [1,16], value = [0.0]);
	conv = conv(input, filter, bias, groups = 0);
	output = {name}(conv);
}}
"""


path = sys.argv[1] if len(sys.argv) > 1 else ''


for op in ops:
    with open(os.path.join(path, '{}_2d.nnef'.format(op)), 'w') as file:
        file.write(template_2d.format(name=op))
    with open(os.path.join(path, '{}_4d.nnef'.format(op)), 'w') as file:
        file.write(template_4d.format(name=op))
    with open(os.path.join(path, '{}_linear.nnef'.format(op)), 'w') as file:
        file.write(template_linear.format(name=op))
