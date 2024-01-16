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
        return 'float32'
    if elem_type == 'integer':
        return 'int32'
    if elem_type == 'logical':
        return 'bool'
    if elem_type == 'string':
        return 'string'
    raise TypeError(f'Type \'{elem_type}\' is not implemented')


def dimension_picker(prefix, kernel_shape, suffix=''):
    """
    Returns the correct name for nth dimensional operator. Uses the 'kernel_shape' attribute.\n
    E.g.call: dimension_picker(op_name)(attr)

    :param prefix: the name of the operator (e.g. conv)
    :param kernel_shape: Shape of the tensor to fit the operation
    :param suffix: optional suffix for ops
    :return: 'prefix`n`d' where n is the correct dimension for the kernel
    """

    rank = len(kernel_shape[2:])
    if rank == 1:
        return prefix + "1d" + suffix
    if rank == 2:
        return prefix + "2d" + suffix
    if rank == 3:
        return prefix + "3d" + suffix
    op_name = prefix + "1d/2d/3d"
    msg = f"Only 1D, 2D, and 3D kernels are supported for operator {op_name}."
    raise tvm.error.OpAttributeInvalid(msg)


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


def _size_conv(size):
    if len(size) == 4:
        # if not isinstance(size[0], tuple):
        assert size[0] == 1 and size[1] == 1, 'Incorrect window dimensions, first two dimensions must be 1'
        return size[2], size[3]
    else:
        raise ValueError(f'Unexpected window size, got {len(size)}')


def _stride_conv(stride):
    if len(stride) == 4:
        assert stride[0] == 1 and stride[1] == 1, 'Incorrect stride dimensions, first two dimensions must be 1'
        return stride[2], stride[3]
    if len(stride) == 2:
        return stride
    else:
        raise ValueError(f'Unexpected window size, got {len(stride)}')


def _padding_conv(padding):
    if isinstance(padding[0], (tuple, list)):
        if len(padding) == 2:
            # change UDLR to ULDR padding LC is faster here
            return [x[i] for i in [0, 1] for x in padding]
        if len(padding) == 4:
            assert padding[0] == (0, 0) and padding[1] == (0, 0), ('Incorrect padding. '
                                                                   'Padding on first two dimensions must be 0')
            # itertools is faster than LC bc of splicing
            return list(itertools.chain.from_iterable(zip(padding[2], padding[3])))
    return padding


def make_parameter_span(source_name_list, name_sep="."):
    return name_sep.join(source_name_list)


def __unexpected_attrs(op: str, kwargs: dict) -> None:
    print(f'{op} received unexpected attributes(s), possibly mismatched versions. Attributes(s) ignored:')
    for k, v in kwargs.items():
        print(f'\t{k} := {v}')


# Conversion map, operator functions

def _get_converter_map():
    return {  # Unary
        'copy': copy_relay_converter,  # arithmetic
        'neg': neg_relay_converter,
        'rcp': rcp_relay_converter,
        'exp': exp_relay_converter,
        'log': log_relay_converter,
        'sin': sin_relay_converter,
        'cos': cos_relay_converter,
        'abs': abs_relay_converter,
        'sign': sign_relay_converter,
        'not': not_relay_converter,  # logical
        'floor': floor_relay_converter,  # rounding
        'ceil': ceil_relay_converter,
        'round': round_relay_converter,
        # Binary
        'add': add_relay_converter,  # arithmetic
        'sub': sub_relay_converter,
        'mul': mul_relay_converter,
        'div': div_relay_converter,
        'pow': pow_relay_converter,
        'lt': lt_relay_converter,  # comparison
        'gt': gt_relay_converter,
        'le': le_relay_converter,
        'ge': ge_relay_converter,
        'eq': eq_relay_converter,
        'ne': ne_relay_converter,
        'and': and_relay_converter,  # logical
        'or': or_relay_converter,
        # select
        'select': select_relay_converter,
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
        'conv': conv_relay_converter,
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
        'squeeze': squeeze_relay_converter,
        'unsqueeze': unsqueeze_relay_converter,
        'transpose': ndop,
        'split': ndop,
        'concat': concatenate_relay_converter,
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
        'matmul': matmul_relay_converter,
        # variables
        'update': ndop,
        # Compound
        'sigmoid': ndop,  # activation
        'relu': relu_relay_converter,
        'prelu': ndop,
        'leaky_relu': ndop,
        'elu': ndop,
        'tanh': ndop,
        'softmax': softmax_relay_converter,
        'softplus': ndop,
        'linear': ndop,  # linear
        'separable_conv': ndop,
        'separable_deconv': ndop,
        'max_pool_with_index': ndop,  # pooling
        'max_pool': max_pool_relay_converter,
        'avg_pool': avg_pool_relay_converter,
        'rms_pool': ndop,
        'local_response_normalization': lrn_relay_converter,  # normalization
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


def copy_relay_converter(data,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('copy', kwargs)

    return get_relay_op('copy')(data)


def neg_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('neg', kwargs)

    return get_relay_op('negative')(data)


def rcp_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('rcp', kwargs)

    raise NotImplementedError('There is no equivalent tp rcp in Relay, TODO in cpp side')


def exp_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('exp', kwargs)

    return get_relay_op('exp')(data)


def log_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('log', kwargs)

    return get_relay_op('log')(data)


def sin_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('sin', kwargs)

    return get_relay_op('sin')(data)


def cos_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('cos', kwargs)

    return get_relay_op('cos')(data)


def abs_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('abs', kwargs)

    return get_relay_op('abs')(data)


def sign_relay_converter(data,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('sign', kwargs)

    return get_relay_op('sign')(data)


def not_relay_converter(data,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('not', kwargs)

    return get_relay_op('logical_not')(data)


def floor_relay_converter(data,
                          **kwargs):
    if kwargs:
        __unexpected_attrs('floor', kwargs)

    return get_relay_op('floor')(data)


def ceil_relay_converter(data,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('ceil', kwargs)

    return get_relay_op('ceil')(data)


def round_relay_converter(data,
                          **kwargs):
    if kwargs:
        __unexpected_attrs('round', kwargs)

    return get_relay_op('round')(data)


#   # Binary ops

def add_relay_converter(lhs, rhs,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('add', kwargs)

    return get_relay_op('add')(lhs, rhs)


def sub_relay_converter(lhs, rhs,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('sub', kwargs)

    return get_relay_op('subtract')(lhs, rhs)


def mul_relay_converter(lhs, rhs,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('mul', kwargs)

    return get_relay_op('multiply')(lhs, rhs)


def div_relay_converter(lhs, rhs,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('div', kwargs)

    return get_relay_op('divide')(lhs, rhs)


def pow_relay_converter(lhs, rhs,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('pow', kwargs)

    return get_relay_op('power')(lhs, rhs)


def lt_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('lt', kwargs)

    return get_relay_op('less')(lhs, rhs)


def gt_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('gt', kwargs)

    return get_relay_op('greater')(lhs, rhs)


def le_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('le', kwargs)

    return get_relay_op('less_equal')(lhs, rhs)


def ge_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('ge', kwargs)

    return get_relay_op('greater_equal')(lhs, rhs)


def eq_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('eq', kwargs)

    return get_relay_op('equal')(lhs, rhs)


def ne_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('ne', kwargs)

    return get_relay_op('not_equal')(lhs, rhs)


def and_relay_converter(lhs, rhs,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('and', kwargs)

    return get_relay_op('logical_and')(lhs, rhs)


def or_relay_converter(lhs, rhs,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('or', kwargs)

    return get_relay_op('logical_or')(lhs, rhs)


#   # Select op

def select_relay_converter(condition, t_val, f_val,
                           **kwargs):
    if kwargs:
        __unexpected_attrs('or', kwargs)

    get_relay_op('where')(condition, t_val, f_val)
    pass


#   # Simplifier ops
# # TODO skipped
#   # Sliding-window ops
def conv_relay_converter(data,
                         kernel,
                         bias,
                         stride,
                         padding,
                         dilation,
                         groups,
                         border):
    if border != 'constant':
        print(f'Currently {border} border is not supported, used `constant` border')

    kernel_shape = infer_shape(kernel)
    strides = _stride_conv(stride)
    if padding:
        pad = _padding_conv(padding)
    else:
        # TODO NNEF automatic padding calculator
        pad = (0, 0, 1, 1)  # seems like good default but WIP

    channels = kernel_shape[0]

    if not stride:
        strides = (1, 1)

    if not dilation:
        dilation = (1, 1)

    op = get_relay_op(dimension_picker('conv', kernel_shape))
    conv_out = op(
        data=data,
        weight=kernel,
        strides=strides,
        padding=pad,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_shape[2:],
        # #defs
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_layout="",
        out_dtype=""
    )

    res = None
    if isinstance(bias, _expr.Constant):
        if bias.data.numpy() == np.array([0.0]):
            res = conv_out

    if not res:
        # squeeze needed bc nnef has bias [1, channel]
        res = _op.nn.bias_add(conv_out, relay.squeeze(bias, axis=0))

    return res


#   # Reduce ops
#   # Tensor shape ops
def squeeze_relay_converter(data, axes):
    return relay.squeeze(data, axes)


def unsqueeze_relay_converter(data, axes):
    # TODO testing how axes is built up
    axes = sorted(axes)
    for axis in axes:
        if axis < 0 and isinstance(data, _expr.Var):
            axis = len(data.type_annotation.concrete_shape) + len(axes) + axis
        data = _op.expand_dims(data, axis=axis, num_newaxis=1)
    return data


def concatenate_relay_converter(*data, axis):
    return relay.concatenate(data, axis)


#   # Region-of-interest ops
#   # Matrix multiplication
def matmul_relay_converter(a, b, transposeA, transposeB):
    """
    Matmul using `dense` for 2D and `batch_matmul` for higher
    """
    # TODO batch matmul
    # a_shape = infer_shape(a)
    # a_ndims = len(a_shape)
    #
    # b_shape = infer_shape(b)
    # b_ndims = len(b_shape)
    #
    # # TODO? check sizes
    #

    out = _op.nn.matmul(a, b, transpose_a=transposeA, transpose_b=transposeB)

    return out


#   # Variable updates
#   # Compound ops

def relu_relay_converter(data):
    return get_relay_op('relu')(data)


def softmax_relay_converter(data, axes):
    if len(axes) > 1:
        print('Multiple axes not supported, operation has been done along the first axis in axes.')
    axis = axes[0]

    s = infer_shape(data)

    return get_relay_op('softmax')(data, axis)


def max_pool_relay_converter(data,
                             size,
                             border,
                             padding,
                             stride,
                             dilation):
    # attr convs
    pool_size = _size_conv(size)
    strides = _stride_conv(stride)
    pad = _padding_conv(padding)

    op = get_relay_op(dimension_picker('max_pool', infer_shape(data)))
    return op(data,
              pool_size=pool_size,
              strides=strides,
              dilation=dilation,
              padding=pad,
              # #defs
              layout="NCHW",
              out_layout="",
              ceil_mode=False
              )


def avg_pool_relay_converter(data,
                             size,
                             border,
                             padding,
                             stride,
                             dilation):
    pool_size = _size_conv(size)
    strides = _stride_conv(stride)
    pad = _padding_conv(padding)

    op = get_relay_op(dimension_picker('avg_pool', infer_shape(data)))
    return op(data,
              pool_size=pool_size,
              strides=strides,
              dilation=dilation,
              padding=pad,
              # #defs
              layout="NCHW",
              out_layout="",
              ceil_mode=False,
              count_include_pad=False
              )


def lrn_relay_converter(data,
                        size,
                        alpha,
                        beta,
                        bias):
    axis = [i for i in range(len(size)) if size[i] > 1]
    if len(axis) == 1:
        axis = axis[0]
    else:
        print("Multi axis LRN is not implemented properly, using axis = 1")
        axis = 1
    size = size[axis]
    return get_relay_op('lrn')(data,
                               size,
                               axis,
                               bias,
                               alpha,
                               beta)


#   # Misc ops


# Converter class
class NNEF_Converter:

    def __init__(self):
        self._nodes = {}
        self._consts = {}
        self._inputs = {}
        self._num_inputs = 0
        self._params = {}
        self._num_params = 0

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
        func = function.Function(list(self._inputs.values()), outputs)
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
                tens_data = i_tens.data
                self._nodes[i_tens.name] = new_var(i_tens.name, shape=op.attribs['shape'],
                                                   dtype=get_type(i_tens.dtype))
                self._params[i_tens.name] = tens_data
            elif op.name == 'constant':
                self._set_const(op)
            else:
                self._set_literal_inputs(op)
                self._set_parameter_span(op, op.name)
                inputs = []
                for ink, inv in op.inputs.items():  # TODO DONE handle default values, without identifier
                    # Extension for list input parameters
                    if isinstance(inv, list):
                        for i, linv in enumerate(inv):
                            if linv in self._nodes.keys():
                                inputs.append(self._nodes[linv])
                            else:  # handle literal inputs
                                name = f'{op.name}_{ink}_{i}'
                                if name in self._nodes.keys():
                                    inputs.append(self._nodes[name])
                                else:
                                    print(f'Invalid input node for {op.name}')
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
                    dtype = 'float32' if not node.dtype else get_type(node.dtype)
                    self._nodes[f'{node.name}_{k}'] = _expr.const(np.array(v, dtype=dtype))

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
    :return: (mod, params) : (tvm.IRModule, dict of str and tvm.nd.NDArray)
    """
    par = NNEF_Converter()
    model = nnef.load_graph(model_path)
    nnef.infer_shapes(model)
    return par.from_nnef(graph=model)
