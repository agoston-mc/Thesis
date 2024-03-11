import os
import platform
import sys

import pytest
import numpy as np

import nnef
from nnef_tools.execute import NNEFExecutor

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor
from relay.utils.tag_span import _create_span, _set_span, _verify_structural_equal_with_span


def verify_model(
        model_path,
        input_data,
        output_data,
        target,
        device,
        rtol=1e-5,
        atol=1e-5,
):

    mod, params = relay.frontend.from_nnef(model_path)

    with tvm.transform.PassContext(opt_level=3):
        # dev = tvm.device(target, 0)
        executor = relay.create_executor(
            'graph', mod, device=device, target=target, params=params
        ).evaluate()
        out = executor(**input_data)

        if not isinstance(out, (list, tuple)):
            out = [out]

        for i, base_out in enumerate(output_data):
            tvm.testing.assert_allclose(out[i].numpy(), output_data[base_out], rtol=rtol, atol=atol)


graphs_dir = os.path.join('tests', 'python', 'frontend', 'nnef', 'outputs')


def _get_model_paths():
    graphs = os.listdir(graphs_dir)
    return [os.path.join(graphs_dir, g) for g in graphs]


def _read_tensor(filename):
    with open(filename) as file:
        return nnef.read_tensor(file)


@tvm.testing.parametrize_targets('llvm')
def NO__test__all_tests(target, dev):
    models = _get_model_paths()
    for model in models:
        if model in ['tests/python/frontend/nnef/outputs/split_channel',
                     'tests/python/frontend/nnef/graphs/deconv7x7',
                     'tests/python/frontend/nnef/outputs/max_pool3x3_constant-border',
                     ]:
            continue
        if  'argmax_pool' in model: # 'deconv' in model or 'debox' in model or
            continue
        if 'upsample' in model:
            if 'symmetric' in model:
                # if 'constant' in model:
                continue

        if 'conv4x4' in model or 'conv5x5' in model or 'conv6x6' in model or 'conv7x7' in model or 'max_pool3x3' in model:
            atol = 1e-2
            rtol = 1e-4
        else:
            atol = 1e-5
            rtol = 1e-5
        graph = nnef.load_graph(model)
        in_data = {}
        for inp in graph.inputs:
            in_data[inp] = _read_tensor(os.path.join(model, f'{inp}.dat'))
        out_data = {}
        for out in graph.outputs:
            out_data[out] = _read_tensor(os.path.join(model, f'{out}.dat'))
        print(model)
        verify_model(model, in_data, out_data, target, dev, atol=atol, rtol=rtol)


def call_test_case(
        name,
        target='llvm',
        dev=tvm.cpu(0),
        rtol=1e-5,
        atol=1e-5,
):
    models = _get_model_paths()
    for model in models:
        in_data = {}
        in_data[name] = _read_tensor(os.path.join(model, 'input.dat'))
        verify_model(model, in_data, target, dev, atol=atol, rtol=rtol)

# CASES START

@tvm.testing.parametrize_targets
def test_gt_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/gt_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_mean_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/mean_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_select_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/select_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad1_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3_pad1-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_relu(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/relu'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_split_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/split_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_rcp_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/rcp_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_rcp_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/rcp_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_log2_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/log2_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_stride2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_stride2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_lt_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/lt_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_or_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/or_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv7x7(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv7x7'
    atol = 1e-4
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_nearest_upsample(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/nearest_upsample'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ceil_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ceil_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_floor_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/floor_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_log_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/log_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sum_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sum_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_min_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/min_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad0_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3_pad0-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_cos_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/cos_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_not_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/not_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sub_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sub_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_aligned_replicate(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/bilinear_upsample_aligned_replicate'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_log_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/log_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_argmin_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/argmin_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_select_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/select_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ne_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ne_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_or_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/or_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_eq_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/eq_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_rsqr_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/rsqr_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_eq_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/eq_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv7x7_stride4x4(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv7x7_stride4x4'
    atol = 1e-2
    rtol = 1e-3
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))

    calc_bl = np.array(list(NNEFExecutor(case_path, None, None)(inputs)[0].values()))
    print(np.allclose(calc_bl, outputs['output']))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_and_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/and_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_mul_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/mul_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_softmax(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/softmax'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sign_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sign_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_mul_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/mul_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_le_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/le_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_or_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/or_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv5x5(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv5x5'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box3x3_pad1_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box3x3_pad1-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox3x3_pad1_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox3x3_pad1-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ge_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ge_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_linear_reshape(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/linear_reshape'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_le_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/le_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_nearest_downsample(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/nearest_downsample'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_select_4d_true(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/select_4d_true'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_min_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/min_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sum_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sum_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_min_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/min_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ge_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ge_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv4x4_stride2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv4x4_stride2x2'
    atol = 5e-3
    rtol = 1e-4
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_reshape_flatten(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/reshape_flatten'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_nobias(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_nobias'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_tile_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/tile_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_softmax_4d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/softmax_4d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_rsqrt_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/rsqrt_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_concat_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/concat_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_area_downsample(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/area_downsample'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad1_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3_pad1-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sigmoid_2d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sigmoid_2d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ne_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ne_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_all_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/all_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_squeeze_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/squeeze_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_and_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/and_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3_constant_border(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3_constant-border'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_argmax_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/argmax_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_cos_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/cos_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sqr_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sqr_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_rsqrt_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/rsqrt_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_symmetric_replicate(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/bilinear_upsample_symmetric_replicate'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_tile_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/tile_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_div_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/div_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sqrt_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sqrt_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_and_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/and_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_transpose_nhwc_to_nchw(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/transpose_nhwc_to_nchw'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad0_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3_pad0-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_round_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/round_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box3x3_pad0_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box3x3_pad0-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv6x6(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv6x6'
    atol = 1e-4
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_add_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/add_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_lt_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/lt_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_min_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/min_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box3x3_stride1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box3x3_stride1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_linear_nobias(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/linear_nobias'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_div_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/div_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_stride1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3_stride1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv7x7(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv7x7'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_groups0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_groups0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_mul_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/mul_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad1_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_pad1-0'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ne_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ne_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad1_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3_pad1-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_mean_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/mean_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv5x5(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv5x5'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3_stride1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3_stride1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pad_1_0_replicate(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pad_1-0_replicate'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox3x3_pad1_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox3x3_pad1-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad1_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3_pad1-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_symmetric_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/bilinear_upsample_symmetric_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_gt_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/gt_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_tanh_4d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/tanh_4d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_add_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/add_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_rsqr_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/rsqr_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_div_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/div_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_eq_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/eq_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_valid(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_valid'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_min_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/min_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_or_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/or_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_min_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/min_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_asymmetric_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/bilinear_upsample_asymmetric_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_pad0_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_pad0-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_pad1_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_pad1-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_abs_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/abs_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ge_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ge_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_transpose_nchw_to_nhwc(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/transpose_nchw_to_nhwc'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad1_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_pad1-1'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ne_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ne_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sqr_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sqr_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_pad1_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_pad1-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_aligned_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/bilinear_upsample_aligned_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_log2_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/log2_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_slice(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/slice'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv2x2'
    # TODO check
    atol = 1e-1
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_all_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/all_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sqrt_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sqrt_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv7x7_stride4x4(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv7x7_stride4x4'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ge_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ge_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_any_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/any_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_and_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/and_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_add_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/add_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_copy_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/copy_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_ceil_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/ceil_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_linear_squeeze(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/linear_squeeze'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sub_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sub_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_valid(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_valid'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pow_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pow_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pad_1_1_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pad_1-1_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox3x3'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_exp_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/exp_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_ignore_border(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3_ignore-border'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad0_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_pad0-0'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pow_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pow_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_abs_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/abs_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sin_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sin_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_select_2d_true(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/select_2d_true'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_relu_2d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/relu_2d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_reshape_squeeze(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/reshape_squeeze'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sub_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sub_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_linear(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/linear'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pow_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pow_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox3x3_pad0_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox3x3_pad0-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_floor_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/floor_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_nobias(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_nobias'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_batch_norm(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/batch_norm'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_stride2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_stride2x2'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pad_0_1_replicate(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pad_0-1_replicate'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_mul_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/mul_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox3x3_pad0_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox3x3_pad0-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_argmin_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/argmin_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_copy_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/copy_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_not_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/not_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sigmoid_4d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sigmoid_4d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_exp_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/exp_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_lt_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/lt_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv4x4(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv4x4'
    atol = 5e-3
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_avg_pool3x3_pad0_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/avg_pool3x3_pad0-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv3x3_pad0_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv3x3_pad0-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pad_0_1_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pad_0-1_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv4x4(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv4x4'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_neg_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/neg_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_bilinear_upsample_asymmetric_replicate(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/bilinear_upsample_asymmetric_replicate'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv5x5_stride3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv5x5_stride3x3'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_relu_4d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/relu_4d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv5x5_pad2_2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv5x5_pad2-2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_tile_batch(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/tile_batch'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_eq_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/eq_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_lt_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/lt_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv1x1'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sign_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sign_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_select_2d_false(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/select_2d_false'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_div_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/div_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pow_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pow_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_round_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/round_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_debox3x3_stride1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/debox3x3_stride1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv5x5_stride3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv5x5_stride3x3'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sub_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sub_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_any_reduce_spatial(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/any_reduce_spatial'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_gt_4d_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/gt_4d_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv6x6(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv6x6'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_le_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/le_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_gt_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/gt_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv4x4_stride2x2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv4x4_stride2x2'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_le_4d_broadcast(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/le_4d_broadcast'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_tanh_2d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/tanh_2d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box3x3(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box3x3'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_select_4d_false(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/select_4d_false'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_tanh(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/tanh'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sin_2d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sin_2d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box3x3_pad0_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box3x3_pad0-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box1x1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box1x1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_box3x3_pad1_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/box3x3_pad1-1'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_conv5x5_pad2_2(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/conv5x5_pad2-2'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_max_pool3x3_pad0_0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/max_pool3x3_pad0-0'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_softmax_2d_standalone(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/softmax_2d_standalone'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_groups0(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_groups0'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_deconv3x3_pad0_1(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/deconv3x3_pad0-1'
    atol = 1e-2
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_sigmoid(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/sigmoid'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_argmax_reduce_channel(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/argmax_reduce_channel'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pad_1_1_replicate(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pad_1-1_replicate'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_pad_1_0_constant(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/pad_1-0_constant'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_unsqueeze(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/unsqueeze'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_neg_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/neg_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)


@tvm.testing.parametrize_targets
def test_add_4d(target, dev):
    case_path = 'tests/python/frontend/nnef/outputs/add_4d'
    atol = 1e-5
    rtol = 1e-5
    graph = nnef.load_graph(case_path)
    inputs = {}
    outputs = {}
    for inp in graph.inputs:
        inputs[inp] = _read_tensor(os.path.join(case_path, f'{inp}.dat'))
    for out in graph.outputs:
        outputs[out] = _read_tensor(os.path.join(case_path, f'{out}.dat'))
    verify_model(case_path, inputs, outputs, target, dev, rtol, atol)

