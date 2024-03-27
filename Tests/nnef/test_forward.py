import os
import pytest

import numpy as np

import nnef
from nnef_tools.execute import NNEFExecutor

import tvm
import tvm.testing
from tvm import relay

# from tvm.contrib import graph_executor
# from relay.utils.tag_span import _create_span, _set_span, _verify_structural_equal_with_span


graphs_dir = os.path.join('tests', 'python', 'frontend', 'nnef', 'outputs')


def _read_tensor(filename):
    with open(filename) as file:
        return nnef.read_tensor(file)


def get_nnef_outputs(path, inputs):
    return NNEFExecutor(path, None, None)(inputs)[0]


def get_type(val):
    if val == 'scalar':
        return 'float32'
    if val == 'integer':
        return 'int32'
    if val == 'logical':
        return 'bool'
    if val == 'string':
        return 'string'


def verify_model(
        model_path,
        target,
        device,
        rtol=1e-5,
        atol=1e-5,
):
    path = os.path.join(graphs_dir, model_path)
    graph = nnef.load_graph(path)
    nnef.infer_shapes(graph)
    inputs = {}
    for inp in graph.inputs:
        intensor = graph.tensors[inp]
        shape = intensor.shape
        inputs[inp] = np.random.uniform(size=shape).astype(get_type(intensor.dtype))
    outputs = get_nnef_outputs(path, inputs)

    mod, params = relay.frontend.from_nnef(path)

    with tvm.transform.PassContext(opt_level=3):
        # dev = tvm.device(target, 0)
        executor = relay.create_executor(
            'graph', mod, device=device, target=target, params=params
        ).evaluate()
        out = executor(**inputs)

        if not isinstance(out, (list, tuple)):
            out = [out]

        for i, base_out in enumerate(outputs):
            tvm.testing.assert_allclose(out[i].numpy(), outputs[base_out], rtol=rtol, atol=atol)


# graph tests


# GENERATED CASES START

@tvm.testing.parametrize_targets
def test_gt_2d(target, dev):
    verify_model('gt_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_2d(target, dev):
    verify_model('max_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_mean_reduce_spatial(target, dev):
    verify_model('mean_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_select_4d(target, dev):
    verify_model('select_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad1_0(target, dev):
    verify_model('max_pool3x3_pad1-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_relu(target, dev):
    verify_model('relu', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_split_channel(target, dev):
    verify_model('split_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_rcp_4d(target, dev):
    verify_model('rcp_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_pool2x2(target, dev):
    verify_model('max_pool2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool2x2(target, dev):
    verify_model('avg_pool2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_rcp_2d(target, dev):
    verify_model('rcp_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_log2_4d(target, dev):
    verify_model('log2_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_stride2x2(target, dev):
    verify_model('conv3x3_stride2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_lt_4d_constant(target, dev):
    verify_model('lt_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_or_4d(target, dev):
    verify_model('or_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv7x7(target, dev):
    verify_model('deconv7x7', target, dev, rtol=1e-5, atol=1e-4)


@tvm.testing.parametrize_targets
def test_nearest_upsample(target, dev):
    verify_model('nearest_upsample', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ceil_4d(target, dev):
    verify_model('ceil_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_floor_2d(target, dev):
    verify_model('floor_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool1x1(target, dev):
    verify_model('avg_pool1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_log_4d(target, dev):
    verify_model('log_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sum_reduce_channel(target, dev):
    verify_model('sum_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_min_reduce_spatial(target, dev):
    verify_model('min_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_4d_broadcast(target, dev):
    verify_model('max_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad0_1(target, dev):
    verify_model('max_pool3x3_pad0-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cos_2d(target, dev):
    verify_model('cos_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_not_4d(target, dev):
    verify_model('not_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sub_4d(target, dev):
    verify_model('sub_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_aligned_replicate(target, dev):
    verify_model('bilinear_upsample_aligned_replicate', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_log_2d(target, dev):
    verify_model('log_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_argmin_reduce_spatial(target, dev):
    verify_model('argmin_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_select_2d(target, dev):
    verify_model('select_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ne_4d(target, dev):
    verify_model('ne_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_or_2d(target, dev):
    verify_model('or_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_eq_2d(target, dev):
    verify_model('eq_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_rsqr_2d(target, dev):
    verify_model('rsqr_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_eq_4d(target, dev):
    verify_model('eq_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv7x7_stride4x4(target, dev):
    verify_model('deconv7x7_stride4x4', target, dev, rtol=1e-2, atol=1e-2)


@tvm.testing.parametrize_targets
def test_max_pool3x3(target, dev):
    verify_model('max_pool3x3', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_and_4d(target, dev):
    verify_model('and_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_mul_4d(target, dev):
    verify_model('mul_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_softmax(target, dev):
    verify_model('softmax', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sign_4d(target, dev):
    verify_model('sign_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_mul_4d_constant(target, dev):
    verify_model('mul_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_le_4d_constant(target, dev):
    verify_model('le_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box2x2(target, dev):
    verify_model('box2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_or_4d_broadcast(target, dev):
    verify_model('or_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv5x5(target, dev):
    verify_model('deconv5x5', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box3x3_pad1_0(target, dev):
    verify_model('box3x3_pad1-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_debox3x3_pad1_0(target, dev):
    verify_model('debox3x3_pad1-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ge_4d_broadcast(target, dev):
    verify_model('ge_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_linear_reshape(target, dev):
    verify_model('linear_reshape', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_le_2d(target, dev):
    verify_model('le_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3(target, dev):
    verify_model('deconv3x3', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_nearest_downsample(target, dev):
    verify_model('nearest_downsample', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_select_4d_true(target, dev):
    verify_model('select_4d_true', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_min_4d_broadcast(target, dev):
    verify_model('min_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_4d(target, dev):
    verify_model('max_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_4d_constant(target, dev):
    verify_model('max_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sum_reduce_spatial(target, dev):
    verify_model('sum_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_min_2d(target, dev):
    verify_model('min_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ge_2d(target, dev):
    verify_model('ge_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv2x2(target, dev):
    verify_model('conv2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv4x4_stride2x2(target, dev):
    verify_model('conv4x4_stride2x2', target, dev, rtol=1e-4, atol=5e-3)


@tvm.testing.parametrize_targets
def test_debox1x1(target, dev):
    verify_model('debox1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_reshape_flatten(target, dev):
    verify_model('reshape_flatten', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_nobias(target, dev):
    verify_model('conv3x3_nobias', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_tile_spatial(target, dev):
    verify_model('tile_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_softmax_4d_standalone(target, dev):
    verify_model('softmax_4d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_rsqrt_4d(target, dev):
    verify_model('rsqrt_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_concat_channel(target, dev):
    verify_model('concat_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_area_downsample(target, dev):
    verify_model('area_downsample', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad1_1(target, dev):
    verify_model('max_pool3x3_pad1-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sigmoid_2d_standalone(target, dev):
    verify_model('sigmoid_2d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ne_4d_constant(target, dev):
    verify_model('ne_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3(target, dev):
    verify_model('conv3x3', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_all_reduce_channel(target, dev):
    verify_model('all_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_squeeze_spatial(target, dev):
    verify_model('squeeze_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_and_4d_constant(target, dev):
    verify_model('and_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_pool3x3_constant_border(target, dev):
    verify_model('max_pool3x3_constant-border', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_argmax_reduce_spatial(target, dev):
    verify_model('argmax_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cos_4d(target, dev):
    verify_model('cos_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sqr_4d(target, dev):
    verify_model('sqr_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_rsqrt_2d(target, dev):
    verify_model('rsqrt_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_symmetric_replicate(target, dev):
    verify_model('bilinear_upsample_symmetric_replicate', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_tile_channel(target, dev):
    verify_model('tile_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_div_4d(target, dev):
    verify_model('div_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sqrt_2d(target, dev):
    verify_model('sqrt_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_and_4d_broadcast(target, dev):
    verify_model('and_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_transpose_nhwc_to_nchw(target, dev):
    verify_model('transpose_nhwc_to_nchw', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad0_1(target, dev):
    verify_model('avg_pool3x3_pad0-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_round_2d(target, dev):
    verify_model('round_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box3x3_pad0_1(target, dev):
    verify_model('box3x3_pad0-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv6x6(target, dev):
    verify_model('deconv6x6', target, dev, rtol=1e-5, atol=1e-4)


@tvm.testing.parametrize_targets
def test_add_4d_constant(target, dev):
    verify_model('add_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_lt_2d(target, dev):
    verify_model('lt_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_min_4d(target, dev):
    verify_model('min_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box3x3_stride1x1(target, dev):
    verify_model('box3x3_stride1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_linear_nobias(target, dev):
    verify_model('linear_nobias', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_div_2d(target, dev):
    verify_model('div_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_stride1x1(target, dev):
    verify_model('avg_pool3x3_stride1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv7x7(target, dev):
    verify_model('conv7x7', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_conv3x3_groups0(target, dev):
    verify_model('conv3x3_groups0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_mul_2d(target, dev):
    verify_model('mul_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad1_0(target, dev):
    verify_model('deconv3x3_pad1-0', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_ne_2d(target, dev):
    verify_model('ne_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad1_1(target, dev):
    verify_model('avg_pool3x3_pad1-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_mean_reduce_channel(target, dev):
    verify_model('mean_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv5x5(target, dev):
    verify_model('conv5x5', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_max_pool3x3_stride1x1(target, dev):
    verify_model('max_pool3x3_stride1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pad_1_0_replicate(target, dev):
    verify_model('pad_1-0_replicate', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_debox3x3_pad1_1(target, dev):
    verify_model('debox3x3_pad1-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad1_0(target, dev):
    verify_model('avg_pool3x3_pad1-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_symmetric_constant(target, dev):
    verify_model('bilinear_upsample_symmetric_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_gt_4d_broadcast(target, dev):
    verify_model('gt_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_tanh_4d_standalone(target, dev):
    verify_model('tanh_4d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_add_2d(target, dev):
    verify_model('add_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_rsqr_4d(target, dev):
    verify_model('rsqr_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_div_4d_broadcast(target, dev):
    verify_model('div_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_eq_4d_broadcast(target, dev):
    verify_model('eq_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_valid(target, dev):
    verify_model('conv3x3_valid', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_min_4d_constant(target, dev):
    verify_model('min_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_or_4d_constant(target, dev):
    verify_model('or_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_min_reduce_channel(target, dev):
    verify_model('min_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_reduce_spatial(target, dev):
    verify_model('max_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_asymmetric_constant(target, dev):
    verify_model('bilinear_upsample_asymmetric_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_pad0_0(target, dev):
    verify_model('conv3x3_pad0-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_pad1_0(target, dev):
    verify_model('conv3x3_pad1-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_abs_2d(target, dev):
    verify_model('abs_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_reduce_channel(target, dev):
    verify_model('max_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ge_4d_constant(target, dev):
    verify_model('ge_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_transpose_nchw_to_nhwc(target, dev):
    verify_model('transpose_nchw_to_nhwc', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad1_1(target, dev):
    verify_model('deconv3x3_pad1-1', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_ne_4d_broadcast(target, dev):
    verify_model('ne_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sqr_2d(target, dev):
    verify_model('sqr_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_pad1_1(target, dev):
    verify_model('conv3x3_pad1-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_aligned_constant(target, dev):
    verify_model('bilinear_upsample_aligned_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_log2_2d(target, dev):
    verify_model('log2_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_slice(target, dev):
    verify_model('slice', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv2x2(target, dev):
    verify_model('deconv2x2', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_all_reduce_spatial(target, dev):
    verify_model('all_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sqrt_4d(target, dev):
    verify_model('sqrt_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv7x7_stride4x4(target, dev):
    verify_model('conv7x7_stride4x4', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_ge_4d(target, dev):
    verify_model('ge_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_any_reduce_channel(target, dev):
    verify_model('any_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_and_2d(target, dev):
    verify_model('and_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_add_4d_broadcast(target, dev):
    verify_model('add_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_copy_2d(target, dev):
    verify_model('copy_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ceil_2d(target, dev):
    verify_model('ceil_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_linear_squeeze(target, dev):
    verify_model('linear_squeeze', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sub_2d(target, dev):
    verify_model('sub_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_valid(target, dev):
    verify_model('deconv3x3_valid', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_pow_4d(target, dev):
    verify_model('pow_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pad_1_1_constant(target, dev):
    verify_model('pad_1-1_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_debox3x3(target, dev):
    verify_model('debox3x3', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv1x1(target, dev):
    verify_model('conv1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_exp_4d(target, dev):
    verify_model('exp_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_ignore_border(target, dev):
    verify_model('avg_pool3x3_ignore-border', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad0_0(target, dev):
    verify_model('deconv3x3_pad0-0', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_pow_4d_broadcast(target, dev):
    verify_model('pow_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_abs_4d(target, dev):
    verify_model('abs_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sin_4d(target, dev):
    verify_model('sin_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_select_2d_true(target, dev):
    verify_model('select_2d_true', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_relu_2d_standalone(target, dev):
    verify_model('relu_2d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_reshape_squeeze(target, dev):
    verify_model('reshape_squeeze', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sub_4d_constant(target, dev):
    verify_model('sub_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_linear(target, dev):
    verify_model('linear', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pow_2d(target, dev):
    verify_model('pow_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_debox3x3_pad0_1(target, dev):
    verify_model('debox3x3_pad0-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_floor_4d(target, dev):
    verify_model('floor_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_nobias(target, dev):
    verify_model('deconv3x3_nobias', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_batch_norm(target, dev):
    verify_model('batch_norm', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_stride2x2(target, dev):
    verify_model('deconv3x3_stride2x2', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_debox2x2(target, dev):
    verify_model('debox2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pad_0_1_replicate(target, dev):
    verify_model('pad_0-1_replicate', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_mul_4d_broadcast(target, dev):
    verify_model('mul_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_debox3x3_pad0_0(target, dev):
    verify_model('debox3x3_pad0-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_argmin_reduce_channel(target, dev):
    verify_model('argmin_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_copy_4d(target, dev):
    verify_model('copy_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_not_2d(target, dev):
    verify_model('not_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sigmoid_4d_standalone(target, dev):
    verify_model('sigmoid_4d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_exp_2d(target, dev):
    verify_model('exp_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_lt_4d(target, dev):
    verify_model('lt_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv4x4(target, dev):
    verify_model('conv4x4', target, dev, rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_avg_pool3x3(target, dev):
    verify_model('avg_pool3x3', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad0_0(target, dev):
    verify_model('avg_pool3x3_pad0-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv3x3_pad0_1(target, dev):
    verify_model('conv3x3_pad0-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pad_0_1_constant(target, dev):
    verify_model('pad_0-1_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv4x4(target, dev):
    verify_model('deconv4x4', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_neg_2d(target, dev):
    verify_model('neg_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_asymmetric_replicate(target, dev):
    verify_model('bilinear_upsample_asymmetric_replicate', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv5x5_stride3x3(target, dev):
    verify_model('conv5x5_stride3x3', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_relu_4d_standalone(target, dev):
    verify_model('relu_4d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_max_pool1x1(target, dev):
    verify_model('max_pool1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv5x5_pad2_2(target, dev):
    verify_model('deconv5x5_pad2-2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_tile_batch(target, dev):
    verify_model('tile_batch', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_eq_4d_constant(target, dev):
    verify_model('eq_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_lt_4d_broadcast(target, dev):
    verify_model('lt_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv1x1(target, dev):
    verify_model('deconv1x1', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_sign_2d(target, dev):
    verify_model('sign_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_select_2d_false(target, dev):
    verify_model('select_2d_false', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_div_4d_constant(target, dev):
    verify_model('div_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pow_4d_constant(target, dev):
    verify_model('pow_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_round_4d(target, dev):
    verify_model('round_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_debox3x3_stride1x1(target, dev):
    verify_model('debox3x3_stride1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv5x5_stride3x3(target, dev):
    verify_model('deconv5x5_stride3x3', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sub_4d_broadcast(target, dev):
    verify_model('sub_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_any_reduce_spatial(target, dev):
    verify_model('any_reduce_spatial', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_gt_4d_constant(target, dev):
    verify_model('gt_4d_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv6x6(target, dev):
    verify_model('conv6x6', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_le_4d(target, dev):
    verify_model('le_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_gt_4d(target, dev):
    verify_model('gt_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv4x4_stride2x2(target, dev):
    verify_model('deconv4x4_stride2x2', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_le_4d_broadcast(target, dev):
    verify_model('le_4d_broadcast', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_tanh_2d_standalone(target, dev):
    verify_model('tanh_2d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box3x3(target, dev):
    verify_model('box3x3', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_select_4d_false(target, dev):
    verify_model('select_4d_false', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_tanh(target, dev):
    verify_model('tanh', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_sin_2d(target, dev):
    verify_model('sin_2d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box3x3_pad0_0(target, dev):
    verify_model('box3x3_pad0-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box1x1(target, dev):
    verify_model('box1x1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_box3x3_pad1_1(target, dev):
    verify_model('box3x3_pad1-1', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_conv5x5_pad2_2(target, dev):
    verify_model('conv5x5_pad2-2', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad0_0(target, dev):
    verify_model('max_pool3x3_pad0-0', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_softmax_2d_standalone(target, dev):
    verify_model('softmax_2d_standalone', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_deconv3x3_groups0(target, dev):
    verify_model('deconv3x3_groups0', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad0_1(target, dev):
    verify_model('deconv3x3_pad0-1', target, dev, rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_sigmoid(target, dev):
    verify_model('sigmoid', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_argmax_reduce_channel(target, dev):
    verify_model('argmax_reduce_channel', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pad_1_1_replicate(target, dev):
    verify_model('pad_1-1_replicate', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_pad_1_0_constant(target, dev):
    verify_model('pad_1-0_constant', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_unsqueeze(target, dev):
    verify_model('unsqueeze', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_neg_4d(target, dev):
    verify_model('neg_4d', target, dev, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_add_4d(target, dev):
    verify_model('add_4d', target, dev, rtol=1e-5, atol=1e-5)
