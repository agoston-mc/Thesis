import math
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
    import (new_var, get_relay_op, fold_constant, set_span, infer_shape, autopad)


# infer_type
# from tvm.topi import get_const_tuple


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
    :param kernel_shape: shape of the tensor to fit the operation
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


def _size_conv(size, rank):
    # window of size (DH)W is only possible when it is checked outside, which is needed for alternative solution
    if rank == 3:
        if len(size) == 1:
            return size
        if len(size) == 3:
            assert size[0] == 1 and size[1] == 1, 'Incorrect window dimensions, first two dimensions must be 1'
            return size[2]
    if rank == 4:
        if len(size) == 2:
            return size
        if len(size) == 4:
            assert size[0] == 1 and size[1] == 1, 'Incorrect window dimensions, first two dimensions must be 1'
            return size[2:]
    if rank == 5:
        if len(size) == 3:
            return size
        if len(size) == 5:
            assert size[0] == 1 and size[1] == 1, 'Incorrect window dimensions, first two dimensions must be 1'
            return size[2:]

    else:
        raise ValueError(f'Unexpected window size, got {len(size)}')


def _stride_conv(stride, rank):
    if rank == 3:
        # {conv style} :: [s] -> [s]
        if len(stride) == 1:
            return stride
        # {pool style} :: [N, C, s] -> asrt N,C == 1; [s]
        if len(stride) == 3:
            assert stride[0] == 1 and stride[1] == 1, 'Not supported stride dimensions, first two dimensions must be 1'
            return stride[2:]
    if rank == 4:
        # {conv style} :: [sh, sw] -> [sh, sw]
        if len(stride) == 2:
            return stride
        # {pool style} :: [N, C, sh, sw] -> asrt N,C == 1; [sh, sw]
        if len(stride) == 4:
            assert stride[0] == 1 and stride[1] == 1, 'Not supported stride dimensions, first two dimensions must be 1'
            return stride[2:]
    if rank == 5:
        # {conv style} :: [sd, sh, sw] -> [sd, sh, sw]
        if len(stride) == 3:
            return stride
        # {pool style} :: [N, C, sd, sh, sw] -> asrt N,C == 1; [sd, sh, sw]
        if len(stride) == 5:
            assert stride[0] == 1 and stride[1] == 1, 'Not supported stride dimensions, first two dimensions must be 1'
            return stride[2:]
    raise ValueError(f'Unexpected stride in {rank - 2}D, got {len(stride)}: {stride}')


def _padding_conv(padding, rank, keepdims=False):
    if isinstance(padding[0], (tuple, list)):
        # 1D
        if rank == 3:
            # {conv style} :: [(l,r)] -> (l,r)
            if len(padding) == 1:
                return padding[0]
            if len(padding) == 3:
                # {pool style} :: [(batch),(channel),(l,r)] -> asrt N,C == 0, (l,r)
                if not keepdims:
                    assert padding[0] == (0, 0) and padding[1] == (0, 0), ('Incorrect padding. '
                                                                           'Padding on C,I dimensions not supported')
                    return padding[2]
                # {sliding window style} :: [(batch),(channel),(l,r)] -> [(batch),(channel),(l,r)]
                else:
                    return padding

                    # 2D

        if rank == 4:
            # {conv style} :: [(u,d),(l,r)] -> (u, l, d, r)
            if len(padding) == 2:
                # change UDLR to ULDR padding, LC is faster here
                return [x[i] for i in [0, 1] for x in padding]

            if len(padding) == 4:
                # {pool style} :: [(batch size),(channel),(u,d),(l,r)] -> asrt N,C == 0, (u, l, d, r)
                if not keepdims:
                    assert padding[0] == (0, 0) and padding[1] == (0, 0), ('Incorrect padding. '
                                                                           'Padding on C,I dimensions not supported')
                    # itertools is faster than LC (slicing)
                    return list(itertools.chain.from_iterable(zip(padding[2], padding[3])))
                # {sliding window style} :: [(batch),(channel),(u,d),(l,r)] -> [(batch),(channel),(u,d),(l,r)]
                else:
                    return padding

                    # 3D

        if rank == 5:
            # {conv style} :: [(f,b),(u,d),(l,r)] -> (f, u, l, b, d, r)
            if len(padding) == 3:
                # LC is faster
                return [x[i] for i in [0, 1] for x in padding]

            if len(padding) == 5:
                # {pool style} :: [(batch size),(channel),(f,b)(u,p),(l,r)] -> asrt N,C == 0, (f, u, l, b, d, r)
                if not keepdims:
                    assert padding[0] == (0, 0) and padding[1] == (0, 0), ('Incorrect padding. '
                                                                           'Padding on C,I dimensions not supported')
                    # itertools faster barely
                    return list(itertools.chain.from_iterable(zip(padding[2], padding[3], padding[4])))
                # {sliding window style} :: [(batch),(channel),(f,b),(u,d),(l,r)] -> [(batch),(channel),(f,b),(u,d),(l,r)]
                else:
                    return padding

        raise ValueError(f'Incorrect padding style for {rank - 2}D operand. Only length of {rank - 2}, {rank} '
                         f'supported, got {len(padding)}: {padding}')

    raise ValueError('nnef should not have singular padding')
    # return padding


def make_parameter_span(source_name_list, name_sep="."):
    return name_sep.join(source_name_list)


def __unexpected_attrs(op, kwargs):
    print(f'{op} received unexpected attributes(s), possibly mismatched versions. Attributes(s) ignored:')
    for k, v in kwargs.items():
        print(f'\t{k} := {v}')


# Conversion map, operator functions

def _get_converter_map():
    return {  # Unary
        'copy': copy_converter,  # arithmetic
        'neg': neg_converter,
        'rcp': rcp_converter,
        'exp': exp_converter,
        'log': log_converter,
        'sin': sin_converter,
        'cos': cos_converter,
        'abs': abs_converter,
        'sign': sign_converter,
        'not': not_converter,  # logical
        'floor': floor_converter,  # rounding
        'ceil': ceil_converter,
        'round': round_converter,
        # Binary
        'add': add_converter,  # arithmetic
        'sub': sub_converter,
        'mul': mul_converter,
        'div': div_converter,
        'pow': pow_converter,
        'lt': lt_converter,  # comparison
        'gt': gt_converter,
        'le': le_converter,
        'ge': ge_converter,
        'eq': eq_converter,
        'ne': ne_converter,
        'and': and_converter,  # logical
        'or': or_converter,
        # select
        'select': select_converter,
        # simplifier
        'sqr': sqr_converter,
        'sqrt': sqrt_converter,
        'rsqr': rsqr_converter,
        'rsqrt': rsqrt_converter,
        'log2': log2_converter,
        'min': min_converter,
        'max': max_converter,
        'clamp': clamp_converter,
        # sliding-window
        'conv': conv_converter,
        'deconv': deconv_converter,
        'box': box_converter,
        'debox': ndop,
        'argmax_pool': argmax_pool_converter,  # ----
        'sample': ndop,
        'desample': ndop,
        'nearest_downsample': ndop,  # barmi pool csak arg mas
        'area_downsample': ndop,  # avg pool kb box, de a stride dil no a factorral
        'nearest_upsample': nearest_upsample_converter,
        'multilinear_upsample': multilinear_upsample_converter,
        # reduce
        'sum_reduce': sum_reduce_converter,
        'max_reduce': max_reduce_converter,
        'min_reduce': min_reduce_converter,
        'argmax_reduce': argmax_reduce_converter,
        'argmin_reduce': argmin_reduce_converter,
        'all_reduce': all_reduce_converter,
        'any_reduce': any_reduce_converter,
        'mean_reduce': mean_reduce_converter,
        # tensor shape
        'reshape': reshape_converter,
        'squeeze': squeeze_converter,
        'unsqueeze': unsqueeze_converter,
        'transpose': transpose_converter,
        'split': split_converter,
        'concat': concat_converter,
        'stack': stack_covnerter,
        'unstack': ndop,
        'slice': slice_converter,
        'pad': pad_converter,
        'tile': tile_converter,
        # region-of-interest
        'avg_roi_pool': ndop,
        'max_roi_pool': ndop,
        'roi_resample': ndop,
        'avg_roi_align': ndop,
        'max_roi_align': ndop,
        # matrix multiplication
        'matmul': matmul_converter,
        # variables
        'update': ndop,  # ---
        # Compound
        'sigmoid': sigmoid_converter,  # activation
        'relu': relu_converter,
        'prelu': prelu_converter,
        'leaky_relu': leaky_relu_converter,
        'elu': ndop,
        'tanh': tanh_converter,
        'softmax': softmax_converter,
        'softplus': ndop,
        'linear': linear_converter,  # linear
        'separable_conv': ndop,
        'separable_deconv': ndop,
        'max_pool_with_index': ndop,  # pooling
        'max_pool': max_pool_converter,
        'avg_pool': avg_pool_converter,
        'rms_pool': rms_pool,
        'local_response_normalization': lrn_converter,  # normalization
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
    print(args, kwargs)
    raise NotImplementedError


#   # Unary ops


def copy_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('copy', kwargs)

    return get_relay_op('copy')(data)


def neg_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('neg', kwargs)

    return get_relay_op('negative')(data)


def rcp_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('rcp', kwargs)

    raise NotImplementedError('There is no equivalent tp rcp in Relay, TODO in cpp side maybe? ')


def exp_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('exp', kwargs)

    return get_relay_op('exp')(data)


def log_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('log', kwargs)

    return get_relay_op('log')(data)


def sin_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('sin', kwargs)

    return get_relay_op('sin')(data)


def cos_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('cos', kwargs)

    return get_relay_op('cos')(data)


def abs_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('abs', kwargs)

    return get_relay_op('abs')(data)


def sign_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('sign', kwargs)

    return get_relay_op('sign')(data)


def not_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('not', kwargs)

    return get_relay_op('logical_not')(data)


def floor_converter(data,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('floor', kwargs)

    return get_relay_op('floor')(data)


def ceil_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('ceil', kwargs)

    return get_relay_op('ceil')(data)


def round_converter(data,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('round', kwargs)

    return get_relay_op('round')(data)


#   # Binary ops

def add_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('add', kwargs)

    return get_relay_op('add')(lhs, rhs)


def sub_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('sub', kwargs)

    return get_relay_op('subtract')(lhs, rhs)


def mul_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('mul', kwargs)

    return get_relay_op('multiply')(lhs, rhs)


def div_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('div', kwargs)

    return get_relay_op('divide')(lhs, rhs)


def pow_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('pow', kwargs)

    return get_relay_op('power')(lhs, rhs)


def lt_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('lt', kwargs)

    return get_relay_op('less')(lhs, rhs)


def gt_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('gt', kwargs)

    return get_relay_op('greater')(lhs, rhs)


def le_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('le', kwargs)

    return get_relay_op('less_equal')(lhs, rhs)


def ge_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('ge', kwargs)

    return get_relay_op('greater_equal')(lhs, rhs)


def eq_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('eq', kwargs)

    return get_relay_op('equal')(lhs, rhs)


def ne_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('ne', kwargs)

    return get_relay_op('not_equal')(lhs, rhs)


def and_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('and', kwargs)

    return get_relay_op('logical_and')(lhs, rhs)


def or_converter(lhs, rhs,
                 **kwargs):
    if kwargs:
        __unexpected_attrs('or', kwargs)

    return get_relay_op('logical_or')(lhs, rhs)


#   # Select op

def select_converter(condition, t_val, f_val,
                     **kwargs):
    if kwargs:
        __unexpected_attrs('select', kwargs)

    return get_relay_op('where')(condition, t_val, f_val)


#   # Simplifier ops

def sqr_converter(data,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('sqr', kwargs)

    return get_relay_op('power')(data, _expr.const(2.0))  # TODO! check


def sqrt_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('sqrt', kwargs)

    return get_relay_op('sqrt')(data)


def rsqr_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('rsqr', kwargs)

    return get_relay_op('power')(data, _expr.const(-2.0))  # TODO! check


def rsqrt_converter(data,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('rsqrt', kwargs)

    return get_relay_op('rsqrt')(data)


def log2_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('log2', kwargs)

    return get_relay_op('log2')(data)


def min_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('min', kwargs)

    return get_relay_op('minimum')(lhs, rhs)


def max_converter(lhs, rhs,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('max', kwargs)

    return get_relay_op('maximum')(lhs, rhs)


def clamp_converter(x, a, b,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('max', kwargs)

    return get_relay_op('clip')(x, b, a)


#   # Sliding-window ops
def conv_converter(data,
                   kernel,
                   bias,
                   border,
                   stride,
                   padding,
                   dilation,
                   groups,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('conv', kwargs)

    if border != 'constant':
        print(f'Currently {border} border is not supported, used `constant` border')

    kernel_shape = infer_shape(kernel)

    strides = _stride_conv(stride, len(kernel_shape)) if stride \
        else (1,) * (len(kernel_shape) - 2)

    dilation = dilation if dilation else (
            (1,) * (len(kernel_shape) - 2))

    if padding:
        pad = _padding_conv(padding, len(kernel_shape))
    else:
        pad = (0,) * (len(kernel_shape) - 2)
        data = autopad(data,
                       strides,
                       kernel_shape[2:],
                       dilation,
                       # mode ?? == SAME UPPER currently seems fine
                       )
        # autopad seems equal to nnef autopadding equation

    channels = kernel_shape[0]

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
        # #defaults for 2d
        # data_layout="NCHW",
        # kernel_layout="OIHW",
        # out_layout="",
        # out_dtype=""
    )

    res = None
    if isinstance(bias, _expr.Constant):
        if bias.data.numpy() == np.array([0.0]):
            res = conv_out

    if not res:
        # squeeze needed bc nnef has bias of shape [1, channel]
        res = _op.nn.bias_add(conv_out, relay.squeeze(bias, axis=0))

    return res


def deconv_converter(data,
                     kernel,
                     bias,
                     border,
                     stride,
                     padding,
                     dilation,
                     output_shape,
                     groups,
                     **kwargs):
    if kwargs:
        __unexpected_attrs('deconv', kwargs)

    pass  # TODO can this be equal to conv transpose?

    if border != 'constant':
        print(f'Currently {border} border is not supported, used `constant` border')

    kernel_shape = infer_shape(kernel)

    rank = len(kernel_shape)

    strides = _stride_conv(stride, rank) if stride \
        else (1,) * (rank - 2)

    dilation = dilation if dilation else (
            (1,) * (rank - 2))

    if padding:
        pad = _padding_conv(padding, rank)
    else:
        # autopad is not usable here, manually calculate the padding
        data_sh = infer_shape(data)[2:]
        out_sh = output_shape[2:]  # [(ui + (s - 1)) // s for ui, s in zip(data_sh, strides)]
        dilated = [(f - 1) * d + 1 for f, d in zip(kernel_shape[2:], dilation)]
        total = [max(0, (di - 1) * s + df - ui) for di, s, df, ui in
                 zip(out_sh, strides, dilated, data_sh)]

        pad = _padding_conv([(pad // 2, (pad + 1) // 2) for pad in total], rank)

    channels = kernel_shape[0]

    # TODO convert output shape to output layout+padding

    op = get_relay_op(dimension_picker('conv', kernel_shape, suffix='_transpose'))
    deconv_out = op(
        data=data,
        weight=kernel,
        strides=strides,
        padding=pad,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_shape[2:],
        # #defaults for 2d
        # data_layout="NCHW",
        # kernel_layout="OIHW",
        # out_layout="",
        # out_dtype=""
    )

    res = None
    if isinstance(bias, _expr.Constant):
        if bias.data.numpy() == np.array([0.0]):
            res = deconv_out

    if not res:
        # squeeze needed bc nnef has bias of shape [1, channel]
        res = _op.nn.bias_add(deconv_out, relay.squeeze(bias, axis=0))

    return res


# TODO box debox are what? can they be found in tvm? are they needed?

# box/debox test, probably not efficient
def box_converter(data,
                  size,
                  border,
                  padding,
                  stride,
                  dilation,
                  normalize,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('box', kwargs)

    dshape = infer_shape(data)

    # check if window size is 1 on N, C, avg pool only supports window on D H W
    if size[:2] == [1, 1]:
        out = avg_pool_converter(data, size[2:], border, padding, stride, dilation)
        if not normalize:
            out = mul_converter(out, _expr.const(math.prod(size), dtype='float32'))
        return out

    strides = stride if stride \
        else (1,) * len(dshape)

    dilation = dilation if dilation \
        else (1,) * len(dshape)

    if padding:
        pad = padding
        _padding_conv(padding, len(dshape))
        data = pad_converter(data, pad, border, _expr.const(0.0, 'float32'))
    else:
        # pad = (0,) * (len(dshape) - 2)
        data = autopad(data,
                       strides,
                       dshape[2:],
                       dilation)

    leave_out_dims = len([x for x in size if x == 1])

    # generate widows
    sw = get_relay_op('sliding_window')(data, leave_out_dims, size[leave_out_dims:], strides[leave_out_dims:])
    # collapse generated windows that are over the dim of the input - the ones we need to sum
    axes = [len(dshape) + x for x in range(len(dshape) - 2)]

    # L2 normalize in sum_reduce is not good, so define own
    out = sum_reduce_converter(sw, axes, False, keepdims=False)
    if normalize:
        rhs = _expr.const(np.full([infer_shape(out)], math.prod(size), dtype='float32'))
        out = get_relay_op('divide')(out, rhs)
    return out


def debox_converter(data,
                    size,
                    border,
                    padding,
                    stride,
                    dilation,
                    normalize,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('debox', kwargs)

    return ndop


def argmax_pool_converter(data,
                          size,
                          border,
                          padding,
                          stride,
                          dilation,
                          **kwargs):
    if kwargs:
        __unexpected_attrs('argmax_pool', kwargs)

    raise NotImplementedError('argmax_pool not implemented')
    # TODO maybe do it with slicing?


def sample_converter(data,
                     index,
                     size,
                     border,
                     padding,
                     stride,
                     dilation,
                     **kwargs):
    if kwargs:
        __unexpected_attrs('sample', kwargs)

    shape = infer_shape(data)
    rank = len(shape)

    return ndop


def nearest_upsample_converter(data,
                               factor,
                               **kwargs):
    if kwargs:
        __unexpected_attrs('nearest_upsample', kwargs)

    rank = len(infer_shape(data))

    if rank == 3:
        raise tvm.error.OpError('Upsampling on 1D tensor is not supported by TVM')
    if rank == 4:
        return get_relay_op('upsampling')(data, factor[0], factor[1], method='nearest_neighbor')
    if rank == 5:
        return get_relay_op('upsampling3d')(data, factor[0], factor[1], factor[2], method='nearest_neighbor',
                                            coordinate_transformation_mode='asymmetric')

    raise ValueError('sth very wrong')


def multilinear_upsample_converter(data,
                                   factor,
                                   method,
                                   border,
                                   **kwargs):
    if kwargs:
        __unexpected_attrs('linear_upsample', kwargs)

    # TODO method - border stuff ...
    rank = len(infer_shape(data))

    if rank == 3:
        raise tvm.error.OpError('Upsampling on 1D tensor is not supported by TVM')
    if rank == 4:
        return get_relay_op('upsampling')(data, factor[0], factor[1], method='bilinear')
    if rank == 5:
        return get_relay_op('upsampling3d')(data, factor[0], factor[1], factor[2], method='trilinear')

    raise ValueError('sth very wrong')


#   # Reduce ops

def sum_reduce_converter(data,
                         axes,
                         normalize,
                         keepdims=True,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('sum_reduce', kwargs)

    out = get_relay_op('sum')(data, axes, keepdims=keepdims)
    if normalize:
        # TODO?? ask normalization value epsilon?
        return get_relay_op('l2_normalize')(out, 0, [x - 2 for x in axes])
    return out


def max_reduce_converter(data,
                         axes,
                         keepdims=True,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('max_reduce', kwargs)

    return get_relay_op('max')(data, axes, keepdims=keepdims)


def min_reduce_converter(data,
                         axes,
                         keepdims=True,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('min_reduce', kwargs)

    return get_relay_op('min')(data, axes, keepdims=keepdims)


def argmax_reduce_converter(data,
                            axes,
                            keepdims=True,
                            **kwargs):
    if kwargs:
        __unexpected_attrs('argmax_reduce', kwargs)

    return get_relay_op('argmax')(data, axes, keepdims=keepdims)


def argmin_reduce_converter(data,
                            axes,
                            keepdims=True,
                            **kwargs):
    if kwargs:
        __unexpected_attrs('argmin_reduce', kwargs)

    return get_relay_op('argmin')(data, axes, keepdims=keepdims)


def all_reduce_converter(data,
                         axes,
                         keepdims=True,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('all_reduce', kwargs)

    return get_relay_op('all')(data, axes, keepdims=keepdims)


def any_reduce_converter(data,
                         axes,
                         keepdims=True,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('any_reduce', kwargs)

    return get_relay_op('any')(data, axes, keepdims=keepdims)


def mean_reduce_converter(data,
                          axes,
                          keepdims=True,
                          **kwargs):
    if kwargs:
        __unexpected_attrs('mean_reduce', kwargs)

    return get_relay_op('mean')(data, axes, keepdims=keepdims)


#   # Tensor shape ops

def reshape_converter(data,
                      shape,
                      axis_start,
                      axis_count,
                      **kwargs):
    if kwargs:
        __unexpected_attrs('reshape', kwargs)

    dshape = list(infer_shape(data))
    if axis_count == -1:
        newshape = dshape[:axis_start] + shape
    else:
        suffix_len = len(dshape) - len(shape) - axis_start + 1
        newshape = dshape[:axis_start] + shape + dshape[- suffix_len:]

    return get_relay_op('reshape')(data, newshape)


def squeeze_converter(data, axes,
                      **kwargs):
    if kwargs:
        __unexpected_attrs('squeeze', kwargs)
    return relay.squeeze(data, axes)


def unsqueeze_converter(data, axes,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('unsqueeze', kwargs)
    # TODO testing how axes is built up
    axes = sorted(axes)
    for axis in axes:
        if axis < 0 and isinstance(data, _expr.Var):
            axis = len(data.type_annotation.concrete_shape) + len(axes) + axis
        data = _op.expand_dims(data, axis=axis, num_newaxis=1)
    return data


def transpose_converter(data,
                        axes,
                        **kwargs):
    if kwargs:
        __unexpected_attrs('transpose', kwargs)

    return get_relay_op('transpose')(data, axes)


def split_converter(data,
                    axis,
                    ratios,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('split', kwargs)

    axis_len = infer_shape(data)[axis]
    rat_mul = axis_len / sum(ratios)
    ratio_list = [(r * rat_mul) for r in ratios]

    s = 0
    indices = []
    for r in ratio_list[:-1]:
        s += r
        # Strinctly needs int ...
        indices.append(int(s))

    return get_relay_op('split')(data, indices, axis)


def concat_converter(*data,
                     axis,
                     **kwargs):
    if kwargs:
        __unexpected_attrs('concat', kwargs)

    return get_relay_op('concatenate')(data, axis)


def stack_covnerter(*data,
                    axis,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('stack', kwargs)

    return get_relay_op('stack')(data, axis)


# TODO unstack


def slice_converter(data,
                    axes,
                    begin,
                    end,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('slice', kwargs)

    # Needs manual stride overwrite because TVM slice breaks at multiple axes,
    # TODO?? check with TVM
    stride = [1] * len(axes)

    return get_relay_op('strided_slice')(data, begin, end, strides=stride, axes=axes)


def pad_converter(data,
                  padding,
                  border,
                  value,
                  **kwargs):
    if kwargs:
        __unexpected_attrs('pad', kwargs)

    if border not in ['constant', 'replicate', 'reflect']:
        print(f'{border} border type is not supported in padding. Assumed constant')
        border = 'constant'
    if border == 'replicate':
        border = 'edge'

    return get_relay_op('pad')(data, padding, value, border)


def tile_converter(data,
                   repeats,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('tile', kwargs)

    return get_relay_op('tile')(data, repeats)


#   # Region-of-interest ops

# TODO roi pools ??


#   # Matrix multiplication
def matmul_converter(a, b, transposeA, transposeB, **kwargs):
    """
    Matmul using `dense` for 2D and `batch_matmul` for higher
    """
    if kwargs:
        __unexpected_attrs('matmul', kwargs)

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


def sigmoid_converter(data,
                      **kwargs):
    if kwargs:
        __unexpected_attrs('sigmoid', kwargs)

    return get_relay_op('sigmoid')(data)


def relu_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('relu', kwargs)

    return get_relay_op('relu')(data)


def prelu_converter(data,
                    alpha,
                    **kwargs):
    if kwargs:
        __unexpected_attrs('prelu', kwargs)

    return get_relay_op('prelu')(data, alpha)


def leaky_relu_converter(data,
                         alpha,
                         **kwargs):
    if kwargs:
        __unexpected_attrs('leaky_relu', kwargs)

    return get_relay_op('leaky_relu')(data, alpha)


# TODO elu


def tanh_converter(data,
                   **kwargs):
    if kwargs:
        __unexpected_attrs('tanh', kwargs)

    return get_relay_op('tanh')(data)


def softmax_converter(data,
                      axes,
                      **kwargs):
    if kwargs:
        __unexpected_attrs('softmax', kwargs)

    if len(axes) > 1:
        print('Multiple axes not supported, operation has been done along the first axis in axes.')
    axis = axes[0]

    return get_relay_op('softmax')(data, axis)


# TODO softplus


#   # linear ops

def linear_converter(data,
                     filter,
                     bias,
                     **kwargs):
    if kwargs:
        __unexpected_attrs('linear', kwargs)

    out = get_relay_op('matmul')(data, filter, transpose_b=True)

    return get_relay_op('add_bias')(out, bias)


# TODO separable conv/deconv

# TODO--- max_pool_with_index == argmax pool sol

def max_pool_converter(data,
                       size,
                       border,
                       padding,
                       stride,
                       dilation,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('max_pool', kwargs)

    if border != 'constant':
        print(f'Currently {border} border is not supported, used `constant` border')

    rank = len(infer_shape(data))
    pool_size = _size_conv(size, rank)
    strides = _stride_conv(stride, rank) if stride \
        else (1,) * (rank - 2)

    dilation = dilation if dilation else (
            (1,) * (rank - 2))

    if padding:
        pad = _padding_conv(padding, rank)
    else:
        pad = (0,) * (rank - 2)     # TODO check if autopad is good here
        data = autopad(data,
                       strides,
                       size[2:],
                       dilation,
                       # mode ?? == SAME UPPER currently seems fine
                       )

    op = get_relay_op(dimension_picker('max_pool', infer_shape(data)))
    return op(data,
              pool_size=pool_size,
              strides=strides,
              dilation=dilation,
              padding=pad,
              )


def avg_pool_converter(data,
                       size,
                       border,
                       padding,
                       stride,
                       dilation,
                       **kwargs):
    if kwargs:
        __unexpected_attrs('avg_pool', kwargs)

    if border != 'constant':
        print(f'Currently {border} border is not supported, used `constant` border')

    rank = len(infer_shape(data))
    pool_size = _size_conv(size, rank)
    strides = _stride_conv(stride, rank) if stride \
        else (1,) * (rank - 2)

    dilation = dilation if dilation else (
            (1,) * (rank - 2))

    if padding:
        pad = _padding_conv(padding, rank)
    else:
        pad = (0,) * (rank - 2)  # TODO check if autopad is good here
        data = autopad(data,
                       strides,
                       size[2:],
                       dilation,
                       # mode ?? == SAME UPPER currently seems fine
                       )

    op = get_relay_op(dimension_picker('avg_pool', infer_shape(data)))
    return op(data,
              pool_size=pool_size,
              strides=strides,
              dilation=dilation,
              padding=pad,
              count_include_pad=True
              )


def rms_pool(data,
             size,
             border,
             padding,
             stride,
             dilation,
             **kwargs):
    if kwargs:
        __unexpected_attrs('rms_pool', kwargs)

    return sqrt_converter(
        avg_pool_converter(
            sqr_converter(data),
            size=size,
            border=border,
            padding=padding,
            stride=stride,
            dilation=dilation))


#   # Normalization

def lrn_converter(data,
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


def local_mean_normalization_converter(data,
                                       size,
                                       **kwargs):
    if kwargs:
        __unexpected_attrs('local_mean_normalization', kwargs)

    mean = get_relay_op('mean')(data)


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
                    if not isinstance(converted, _expr.TupleWrapper):
                        converted = fold_constant(converted)
                    else:
                        converted = fold_constant(converted.astuple())
                else:
                    converted = _expr.TupleWrapper(fold_constant(converted.astuple()), len(converted))

                converted = set_span(converted, op.name)

                if outputs_num == 1:
                    # check if the singular ret val is a list of only one element
                    ret_val = list(op.outputs.values())[0]
                    if isinstance(ret_val, list):
                        self._nodes[ret_val[0]] = converted
                    else:
                        self._nodes[ret_val] = converted
                else:
                    for i, out in zip(range(outputs_num), op.outputs['values']):
                        self._nodes[out] = converted[i]
                    # pass

                    # raise NotImplementedError(f'Multiple outputs are not supported. Raised by {op.name}.')
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
