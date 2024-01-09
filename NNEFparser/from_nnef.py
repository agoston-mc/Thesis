import os
import itertools

import numpy as np

import nnef

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.relay import analysis, function
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.relay.frontend.common \
    import new_var, get_relay_op, fold_constant, set_span, infer_shape, infer_type
from tvm.topi import get_const_tuple


# Base methods

def get_type(elem_type):
    """
    Gives numpy style type for nnef primitive types, uses x32 versions.

    :param elem_type: string, (scalar, integer, logical, string)
    :return: returns numpy dtype equivalent (float32, int32, bool, string)
    """
    if elem_type == 'scalar':
        return 'float64'
    if elem_type == 'integer':
        return 'int64'
    if elem_type == 'logical':
        return 'bool'
    if elem_type == 'string':
        return 'string'
    raise TypeError(f'Type \'{elem_type}\' is not implemented')


def dimension_picker(prefix, suffix=''):
    """
    Returns the correct name for nth dimensional operator. Uses the 'kernel_shape' attribute.\n
    E.g.call: dimension_picker(op_name)(attr)

    :param prefix: the name of the operator (e.g. conv)
    :param suffix: optional suffix for ops
    :return: 'prefix`n`d' where n is the correct dimension for the kernel
    """

    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 1:
            return prefix + "1d" + suffix
        if len(kernel) == 2:
            return prefix + "2d" + suffix
        if len(kernel) == 3:
            return prefix + "3d" + suffix
        op_name = prefix + "1d/2d/3d"
        msg = f"Only 1D, 2D, and 3D kernels are supported for operator {op_name}."
        raise tvm.error.OpAttributeInvalid(msg)

    return _impl


def dimension_constraint():
    """
    Checks whether the kernel is 1/2/3-dimensional. Uses the 'kernel_shape' attribute.\n
    E.g.call: dimension_constraint()(attr)
    :return: True if length of kernel is 1,2 or 3. False otherwise.
    """

    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in [1, 2, 3]:
            return True
        return False

    return _dim_check, "Only 1d, 2d and 3d kernel supported."


def make_parameter_span(source_name_list, name_sep="."):
    return name_sep.join(source_name_list)


# Conversion map, operator functions

def _get_converter_map():
    return {  # Unary
        'copy': ndop,  # arithmetic
        'neg': ndop,
        'rcp': ndop,
        'exp': ndop,
        'log': ndop,
        'sin': ndop,
        'cos': ndop,
        'abs': ndop,
        'sign': ndop,
        'not': ndop,  # logical
        'floor': ndop,  # rounding
        'ceil': ndop,
        'round': ndop,
        # Binary
        'add': add_rel_op,  # arithmetic
        'sub': ndop,
        'mul': mul_rel_op,
        'div': ndop,
        'pow': ndop,
        'lt': ndop,  # comparison
        'gt': ndop,
        'le': ndop,
        'ge': ndop,
        'eq': ndop,
        'ne': ndop,
        'and': ndop,  # logical
        'or': ndop,
        # select
        'select': ndop,
        # simplifier
        'sqr': ndop,
        'sqrt': ndop,
        'rsqr': ndop,
        'rsqrt': ndop,
        'log2': ndop,
        'min': ndop,
        'max': ndop,
        'clamp': ndop,
        # sliding-window
        'conv': conv_rel_op,
        'deconv': ndop,
        'box': ndop,
        'debox': ndop,
        'argmax_pool': ndop,
        'sample': ndop,
        'desample': ndop,
        'nearest_downsample': ndop,
        'area_downsample': ndop,
        'nearest_upsample': ndop,
        'multilinear_upsample': ndop,
        # reduce
        'sum_reduce': ndop,
        'max_reduce': ndop,
        'min_reduce': ndop,
        'argmax_reduce': ndop,
        'argmin_reduce': ndop,
        'all_reduce': ndop,
        'any_reduce': ndop,
        'mean_reduce': ndop,
        # tensor shape
        'reshape': ndop,
        'squeeze': squeeze_op,
        'unsqueeze': unsqueeze_op,
        'transpose': ndop,
        'split': ndop,
        'concat': concatenate_op,
        'stack': ndop,
        'unstack': ndop,
        'slice': ndop,
        'pad': ndop,
        'tile': ndop,
        # region-of-interest
        'avg_roi_pool': ndop,
        'max_roi_pool': ndop,
        'roi_resample': ndop,
        'avg_roi_align': ndop,
        'max_roi_align': ndop,
        # matrix multiplication
        'matmul': matmul_rel_op,
        # variables
        'update': ndop,
        # Compound
        'sigmoid': ndop,  # activation
        'relu': relu_rel_op,
        'prelu': ndop,
        'leaky_relu': ndop,
        'elu': ndop,
        'tanh': ndop,
        'softmax': softmax_op,
        'softplus': ndop,
        'linear': ndop,  # linear
        'separable_conv': ndop,
        'separable_deconv': ndop,
        'max_pool_with_index': ndop,  # pooling
        'max_pool': max_pool_op,
        'avg_pool': avg_pool_op,
        'rms_pool': ndop,
        'local_response_normalization': lrn_op,  # normalization
        'local_mean_normalization': ndop,
        'local_variance_normalization': ndop,
        'local_contrast_normalization': ndop,
        'l1_normalization': ndop,
        'l2_normalization': ndop,
        'batch_normalization': ndop,
        'min_max_linear_quantize': ndop,  # quantization
        'zero_point_linear_quantize': ndop,
        'linear_quantize': ndop,
        'logarithmic_quantize': ndop,
        # MISC
        'copy_n': ndop,
        'add_n': ndop,
        'moments': ndop,
    }


def ndop(*args, **kwargs):  # TODO not implemented ops
    raise NotImplementedError


#   # Unary ops


#   # Binary ops

def add_rel_op(lhs, rhs):
    return get_relay_op('add')(lhs, rhs)


def mul_rel_op(lhs, rhs):
    return get_relay_op('multiply')(lhs, rhs)


#   # Select op
#   # Simplifier ops
#   # Sliding-window ops
def conv_rel_op(data,
                kernel,
                bias,
                stride,
                padding,
                dilation,
                groups,
                border):
    pass  # TODO


#   # Reduce ops
#   # Tensor shape ops
def squeeze_op(data, axes):
    return relay.squeeze(data, axes)


def unsqueeze_op(data, axes):
    # TODO testing how axes is built up
    axes = sorted(axes)
    for axis in axes:
        if axis < 0 and isinstance(data, _expr.Var):
            axis = len(data.type_annotation.concrete_shape) + len(axes) + axis
        data = _op.expand_dims(data, axis=axis, num_newaxis=1)
    return data


def concatenate_op(*data, axis):
    return relay.concatenate(data, axis)


#   # Region-of-interest ops
#   # Matrix multiplication
def matmul_rel_op(a, b):
    """
    Matmul using `dense` for 2D and `batch_matmul` for higher (TODO)
    """
    # a_shape = infer_shape(a)
    # a_ndims = len(a_shape)
    #
    # b_shape = infer_shape(b)
    # b_ndims = len(b_shape)
    #
    # # TODO? check sizes
    #

    out = _op.nn.dense(a, _op.transpose(b))

    return out


#   # Variable updates
#   # Compound ops

def relu_rel_op(data):
    return get_relay_op('relu')(data)


def softmax_op(data, axes):
    if len(axes) > 1:
        print('Multiple axes not supported, operation has been done along the first axis in axes.')
    axis = axes[0]
    return get_relay_op('softmax')(data, axis)


def max_pool_op(data,
                size,
                border,
                padding,
                stride,
                dilation):
    pass


def avg_pool_op(data,
                size,
                border,
                padding,
                stride,
                dilation):
    pass


def lrn_op(data,
           size,
           alpha,
           beta,
           bias):
    pass


#   # Misc ops


# Parser class
class NNEF_Parser:

    def __init__(self, root_dir):
        self._nodes = {}
        self._consts = {}
        self._inputs = {}
        self._num_inputs = 0
        self._params = {}
        self._num_params = 0
        self._rootdir = root_dir

    def from_nnef(self, graph):
        self._parse_inputs(graph)
        self._construct_nodes(graph)

        outputs = [self._nodes[n] for n in graph.outputs]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = analysis.free_vars(outputs)
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params.keys():
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        func = function.Function(self._inputs.values(), outputs)
        return IRModule.from_expr(func), self._params

    def _parse_inputs(self, graph: nnef.Graph):
        for inp in graph.inputs:
            if inp in self._params:  # TODO- maybe deletable
                self._num_params += 1
                self._nodes[inp] = new_var(inp, shape=self._params[inp].shape, dtype=self._params[inp].dtype)
            elif inp in self._nodes:
                continue
            else:
                self._num_inputs += 1
                i_tens = graph.tensors[inp]
                self._nodes[inp] = new_var(inp, shape=i_tens.shape, dtype=get_type(i_tens.dtype))
            self._inputs[inp] = self._nodes[inp]

    def _construct_nodes(self, graph):
        for op in graph.operations:
            if op.name == 'external':
                continue
            if op.name == 'variable':
                # TODO convert params to const, or leave variable (freeze vars switch)
                i_tens = graph.tensors[op.outputs['output']]
                with open(os.path.join(self._rootdir, op.attribs['label'] + '.dat')) as f:
                    tens_data = tvm.runtime.ndarray.array(nnef.read_tensor(f))
                self._nodes[i_tens.name] = new_var(i_tens.name, shape=op.attribs['shape'],
                                                   dtype=get_type(i_tens.dtype))
                self._params[i_tens.name] = tens_data
            elif op.name == 'constant':
                self._set_const(op)
            else:
                self._set_literal_inputs(op)
                self._set_parameter_span(op, op.name)
                inputs = []
                for ink, inv in op.inputs.items():  # TODO handle default values, without identifier
                    # Extension for list input paramters
                    if isinstance(inv, list):
                        for i, linv in enumerate(inv):
                            if linv in self._nodes.keys():
                                inputs.append(self._nodes[linv])
                            else:  # handle literal inputs
                                name = f'{op.name}_{ink}_{i}'
                                if name in self._nodes.keys():
                                    inputs.append(self._nodes[name])
                                else:
                                    print('Invalid input node for op')
                    else:
                        if inv in self._nodes.keys():
                            inputs.append(self._nodes[inv])
                        else:  # handle literal inputs
                            name = f'{op.name}_{ink}'
                            if name in self._nodes.keys():
                                inputs.append(self._nodes[name])
                            else:
                                print('Invalid input node for op')

                converted = self._get_relay_opcall(op.name, inputs, op.attribs)

                if not isinstance(converted, _expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(converted)

                if outputs_num == 1:
                    converted = fold_constant(converted)
                else:
                    converted = _expr.TupleWrapper(fold_constant(converted.astuple()), len(converted))

                converted = set_span(converted, op.name)

                if outputs_num == 1:
                    self._nodes[list(op.outputs.values())[0]] = converted
                else:
                    raise NotImplementedError(f'Multiple outputs are not supported. Raised by {op.name}.')
                    # node_output = None
                    # for k, i in zip(list(node_output), range(len(node_output))):
                    #     self._nodes[k] = op[i]

    def _set_const(self, node):
        name = node.outputs['output']
        data = node.attribs['value']
        shape = node.attribs['shape']
        if shape != [1]:
            data = np.full(shape, data, dtype=get_type(node.dtype))
        else:
            data = np.array(data, dtype=get_type(node.dtype))
        self._consts[name] = _expr.const(data)
        self._nodes[name] = self._consts[name]

    def _set_literal_inputs(self, node):
        for k, v in node.inputs.items():
            if isinstance(v, list):
                for ve in v:
                    if ve not in self._nodes.keys():
                        self._nodes[f'{node.name}_{k}'] = _expr.const(np.array(ve, dtype=get_type(node.dtype)))
            else:
                if v not in self._nodes.keys():
                    self._nodes[f'{node.name}_{k}'] = _expr.const(np.array(v, dtype=get_type(node.dtype)))

    def _set_parameter_span(self, node, node_source_name):
        for k, name in node.inputs.items():
            if isinstance(name, list):
                for n in name:
                    self._set_par_span_helper(node, node_source_name, n)
            else:
                self._set_par_span_helper(node, node_source_name, name)

    def _set_par_span_helper(self, node, node_source_name, name):
        expr = self._nodes.get(name)
        if isinstance(expr, (relay.Var, relay.Constant)):
            if isinstance(expr, relay.Constant):
                if name not in self._consts:
                    name = f'{node.name}_'  # TODO# deleted a {k}
            expr_with_span = set_span(expr, make_parameter_span([node_source_name, name]))
            self._nodes[name] = expr_with_span
            if name in self._inputs:
                self._inputs[name] = expr_with_span

    def _get_relay_opcall(self, name, inputs, attrs):
        conv_map = _get_converter_map()
        if name in conv_map:
            call = conv_map[name](*inputs, **attrs)
        else:
            raise NotImplementedError(f'Operator {name} is not implemented.')
        return call


def from_nnef(
        model_path: os.PathLike | str
):
    """

    :param model_path:
    :return: (mod, params) : (tvm.IRModule, dict of str and tvm.nd.NDArray)
    """
    par = NNEF_Parser(model_path)
    model = nnef.load_graph(os.path.join(model_path, 'graph.nnef'))
    nnef.infer_shapes(model)
    return par.from_nnef(graph=model)
