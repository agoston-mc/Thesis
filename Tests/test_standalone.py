# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os

import numpy as np

import _nnef
import nnef
import nnef_tools.interpreter.pytorch as interpreter

import tvm
import tvm.testing
from tvm import relay

import NNEFConverter


graphs_dir = os.path.join("Tests", "cases")


def get_nnef_outputs(path, inputs):
    ip = interpreter.Interpreter(path, None, None)
    inputs = [inputs[tensor.name] for tensor in ip.input_details()]
    return ip(inputs)


def get_type(val):
    if val == "scalar":
        return "float32"
    if val == "integer":
        return "int32"
    if val == "logical":
        return "bool"
    if val == "string":
        return "string"


def verify_model(
        model_path,
        target='llvm',
        device=tvm.cpu(0),
        rtol=1e-5,
        atol=1e-5,
):
    path = os.path.join(graphs_dir, model_path)
    graph = nnef.load_graph(path, load_variables=False)
    nnef.infer_shapes(graph)
    inputs = {}

    # generate inputs
    for inp in graph.inputs:
        intensor = graph.tensors[inp]
        shape = intensor.shape
        if any(exc in model_path for exc in ["log", "sqrt", "pow", "batch_norm"]):
            low = 0.0
        else:
            low = -1.0
        high = 1.0
        if "acosh" in model_path:
            high = 2.0
            low = 1.0
        if intensor.dtype == "scalar":
            inputs[inp] = np.random.uniform(low=low, high=high, size=shape).astype("float32")
        elif intensor.dtype == "integer":
            inputs[inp] = np.random.randint(0, 64, shape)
        elif intensor.dtype == "logical":
            inputs[inp] = np.random.binomial(1, 0.5, shape).astype("bool")
        elif intensor.dtype == "string":
            inputs[inp] = np.random.uniform(low=low, high=high, size=shape).astype("string")

    # set graph parameters
    for operation in graph.operations:
        if operation.name == "variable":
            tensor_name = operation.outputs["output"]

            shape = operation.attribs["shape"]

            assert operation.dtype == 'scalar', \
                f'variable of type {operation.dtype} is not supported, please update verify_model'

            data = np.random.uniform(low=-1.0, size=shape).astype("float32")

            tensor = graph.tensors[tensor_name]
            graph.tensors[tensor_name] = _nnef.Tensor(
                tensor.name, tensor.dtype, shape, data, tensor.quantization
            )

    outputs = get_nnef_outputs(graph, inputs)

    mod, params = NNEFConverter.from_nnef(graph)

    with tvm.transform.PassContext(opt_level=3):
        # dev = tvm.device(target, 0)
        executor = relay.create_executor(
            "graph", mod, device=device, target=target, params=params
        ).evaluate()
        out = executor(**inputs)

        if not isinstance(out, (list, tuple)):
            out = [out]

        for i, base_out in enumerate(outputs):
            tvm.testing.assert_allclose(out[i].numpy(), outputs[base_out], rtol=rtol, atol=atol)


# graph tests


@tvm.testing.parametrize_targets
def test_ats_tan_2d():
    verify_model("tan_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_tan_4d():
    verify_model("tan_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_sinh_2d():
    verify_model("sinh_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_sinh_4d():
    verify_model("sinh_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_cosh_2d():
    verify_model("cosh_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_cosh_4d():
    verify_model("cosh_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_asin_2d():
    verify_model("asin_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_asin_4d():
    verify_model("asin_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_acos_2d():
    verify_model("acos_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_acos_4d():
    verify_model("acos_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_atan_2d():
    verify_model("atan_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_atan_4d():
    verify_model("atan_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_asinh_2d():
    verify_model("asinh_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_asinh_4d():
    verify_model("asinh_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_acosh_2d():
    verify_model("acosh_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_acosh_4d():
    verify_model("acosh_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_atanh_2d():
    verify_model("atanh_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_atanh_4d():
    verify_model("atanh_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_clamp_2d():
    verify_model("clamp_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_clamp_4d():
    verify_model("clamp_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_clamp_4d_constant():
    verify_model("clamp_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_reshape_partial():
    verify_model("reshape_partial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_split_unbalanced():
    verify_model("split_unbalanced", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_stack():
    verify_model("stack", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_unstack():
    verify_model("unstack", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_slice_strides():
    verify_model("slice_strides", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_matmul_2d():
    verify_model("matmul_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_matmul_2d_transpose():
    verify_model("matmul_2d_transpose", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_matmul_4d():
    verify_model("matmul_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_matmul_4d_transpose():
    verify_model("matmul_4d_transpose", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_prelu():
    verify_model("prelu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_prelu_2d_standalone():
    verify_model("prelu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_prelu_4d_standalone():
    verify_model("prelu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_leaky_relu():
    verify_model("leaky_relu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_leaky_relu_2d_standalone():
    verify_model("leaky_relu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_leaky_relu_4d_standalone():
    verify_model("leaky_relu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_elu():
    verify_model("elu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_elu_2d_standalone():
    verify_model("elu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_elu_4d_standalone():
    verify_model("elu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_selu():
    verify_model("selu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_selu_2d_standalone():
    verify_model("selu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_selu_4d_standalone():
    verify_model("selu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_gelu():
    verify_model("gelu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_gelu_2d_standalone():
    verify_model("gelu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_gelu_4d_standalone():
    verify_model("gelu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_silu():
    verify_model("silu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_silu_2d_standalone():
    verify_model("silu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_silu_4d_standalone():
    verify_model("silu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_softplus():
    verify_model("softplus", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_softplus_2d_standalone():
    verify_model("softplus_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_softplus_4d_standalone():
    verify_model("softplus_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_separable_conv3x3():
    verify_model("separable_conv3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_separable_conv3x3_with_attrs():
    verify_model("separable_conv3x3_with_attrs", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_separable_conv5x5():
    verify_model("separable_conv5x5", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_separable_deconv3x3():
    verify_model("separable_deconv3x3", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_ats_separable_deconv3x3_with_attrs():
    verify_model("separable_deconv3x3_with_attrs", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_ats_separable_deconv5x5():
    verify_model("separable_deconv5x5", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_ats_rms_pool3x3():
    verify_model("rms_pool3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_local_response_normalization():
    verify_model("local_response_normalization", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_local_mean_normalization():
    verify_model("local_mean_normalization", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_local_variance_normalization():
    verify_model("local_variance_normalization", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_local_contrast_normalization():
    verify_model("local_contrast_normalization", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_l1_normalization():
    verify_model("l1_normalization", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_l2_normalization():
    verify_model("l2_normalization", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_pad_0_1_reflect():
    verify_model("pad_0-1_reflect", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_pad_1_0_reflect():
    verify_model("pad_1-0_reflect", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_ats_pad_1_1_reflect():
    verify_model("pad_1-1_reflect", rtol=1e-5, atol=1e-5)


# GENERATED CASES START


@tvm.testing.parametrize_targets
def test_cts_gt_2d():
    verify_model("gt_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_2d():
    verify_model("max_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_mean_reduce_spatial():
    verify_model("mean_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_select_4d():
    verify_model("select_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3_pad1_0():
    verify_model("max_pool3x3_pad1-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_relu():
    verify_model("relu", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_split_channel():
    verify_model("split_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_rcp_4d():
    verify_model("rcp_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_pool2x2():
    verify_model("max_pool2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool2x2():
    verify_model("avg_pool2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_rcp_2d():
    verify_model("rcp_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_log2_4d():
    verify_model("log2_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_stride2x2():
    verify_model("conv3x3_stride2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_lt_4d_constant():
    verify_model("lt_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_or_4d():
    verify_model("or_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv7x7():
    verify_model("deconv7x7", rtol=1e-5, atol=1e-4)


@tvm.testing.parametrize_targets
def test_cts_nearest_upsample():
    verify_model("nearest_upsample", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ceil_4d():
    verify_model("ceil_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_floor_2d():
    verify_model("floor_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool1x1():
    verify_model("avg_pool1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_log_4d():
    verify_model("log_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sum_reduce_channel():
    verify_model("sum_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_min_reduce_spatial():
    verify_model("min_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_4d_broadcast():
    verify_model("max_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3_pad0_1():
    verify_model("max_pool3x3_pad0-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_cos_2d():
    verify_model("cos_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_not_4d():
    verify_model("not_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sub_4d():
    verify_model("sub_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_bilinear_upsample_aligned_replicate():
    verify_model("bilinear_upsample_aligned_replicate", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_log_2d():
    verify_model("log_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_argmin_reduce_spatial():
    verify_model("argmin_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_select_2d():
    verify_model("select_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ne_4d():
    verify_model("ne_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_or_2d():
    verify_model("or_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_eq_2d():
    verify_model("eq_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_rsqr_2d():
    verify_model("rsqr_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_eq_4d():
    verify_model("eq_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv7x7_stride4x4():
    verify_model("deconv7x7_stride4x4", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3():
    verify_model("max_pool3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_and_4d():
    verify_model("and_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_mul_4d():
    verify_model("mul_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_softmax():
    verify_model("softmax", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sign_4d():
    verify_model("sign_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_mul_4d_constant():
    verify_model("mul_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_le_4d_constant():
    verify_model("le_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box2x2():
    verify_model("box2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_or_4d_broadcast():
    verify_model("or_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv5x5():
    verify_model("deconv5x5", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box3x3_pad1_0():
    verify_model("box3x3_pad1-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_debox3x3_pad1_0():
    verify_model("debox3x3_pad1-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ge_4d_broadcast():
    verify_model("ge_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_linear_reshape():
    verify_model("linear_reshape", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_le_2d():
    verify_model("le_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3():
    verify_model("deconv3x3", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_nearest_downsample():
    verify_model("nearest_downsample", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_select_4d_true():
    verify_model("select_4d_true", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_min_4d_broadcast():
    verify_model("min_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_4d():
    verify_model("max_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_4d_constant():
    verify_model("max_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sum_reduce_spatial():
    verify_model("sum_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_min_2d():
    verify_model("min_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ge_2d():
    verify_model("ge_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv2x2():
    verify_model("conv2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv4x4_stride2x2():
    verify_model("conv4x4_stride2x2", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_debox1x1():
    verify_model("debox1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_reshape_flatten():
    verify_model("reshape_flatten", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_nobias():
    verify_model("conv3x3_nobias", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_tile_spatial():
    verify_model("tile_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_softmax_4d_standalone():
    verify_model("softmax_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_rsqrt_4d():
    verify_model("rsqrt_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_concat_channel():
    verify_model("concat_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_area_downsample():
    verify_model("area_downsample", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3_pad1_1():
    verify_model("max_pool3x3_pad1-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sigmoid_2d_standalone():
    verify_model("sigmoid_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ne_4d_constant():
    verify_model("ne_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3():
    verify_model("conv3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_all_reduce_channel():
    verify_model("all_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_squeeze_spatial():
    verify_model("squeeze_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_and_4d_constant():
    verify_model("and_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3_constant_border():
    verify_model("max_pool3x3_constant-border", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_argmax_reduce_spatial():
    verify_model("argmax_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_cos_4d():
    verify_model("cos_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sqr_4d():
    verify_model("sqr_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_rsqrt_2d():
    verify_model("rsqrt_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_bilinear_upsample_symmetric_replicate():
    verify_model("bilinear_upsample_symmetric_replicate", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_tile_channel():
    verify_model("tile_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_div_4d():
    verify_model("div_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sqrt_2d():
    verify_model("sqrt_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_and_4d_broadcast():
    verify_model("and_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_transpose_nhwc_to_nchw():
    verify_model("transpose_nhwc_to_nchw", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3_pad0_1():
    verify_model("avg_pool3x3_pad0-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_round_2d():
    verify_model("round_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box3x3_pad0_1():
    verify_model("box3x3_pad0-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv6x6():
    verify_model("deconv6x6", rtol=1e-5, atol=1e-4)


@tvm.testing.parametrize_targets
def test_cts_add_4d_constant():
    verify_model("add_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_lt_2d():
    verify_model("lt_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_min_4d():
    verify_model("min_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box3x3_stride1x1():
    verify_model("box3x3_stride1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_linear_nobias():
    verify_model("linear_nobias", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_div_2d():
    verify_model("div_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3_stride1x1():
    verify_model("avg_pool3x3_stride1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv7x7():
    verify_model("conv7x7", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_groups0():
    verify_model("conv3x3_groups0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_mul_2d():
    verify_model("mul_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_pad1_0():
    verify_model("deconv3x3_pad1-0", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_ne_2d():
    verify_model("ne_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3_pad1_1():
    verify_model("avg_pool3x3_pad1-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_mean_reduce_channel():
    verify_model("mean_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv5x5():
    verify_model("conv5x5", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3_stride1x1():
    verify_model("max_pool3x3_stride1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pad_1_0_replicate():
    verify_model("pad_1-0_replicate", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_debox3x3_pad1_1():
    verify_model("debox3x3_pad1-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3_pad1_0():
    verify_model("avg_pool3x3_pad1-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_bilinear_upsample_symmetric_constant():
    verify_model("bilinear_upsample_symmetric_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_gt_4d_broadcast():
    verify_model("gt_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_tanh_4d_standalone():
    verify_model("tanh_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_add_2d():
    verify_model("add_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_rsqr_4d():
    verify_model("rsqr_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_div_4d_broadcast():
    verify_model("div_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_eq_4d_broadcast():
    verify_model("eq_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_valid():
    verify_model("conv3x3_valid", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_min_4d_constant():
    verify_model("min_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_or_4d_constant():
    verify_model("or_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_min_reduce_channel():
    verify_model("min_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_reduce_spatial():
    verify_model("max_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_bilinear_upsample_asymmetric_constant():
    verify_model("bilinear_upsample_asymmetric_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_pad0_0():
    verify_model("conv3x3_pad0-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_pad1_0():
    verify_model("conv3x3_pad1-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_abs_2d():
    verify_model("abs_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_reduce_channel():
    verify_model("max_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ge_4d_constant():
    verify_model("ge_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_transpose_nchw_to_nhwc():
    verify_model("transpose_nchw_to_nhwc", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_pad1_1():
    verify_model("deconv3x3_pad1-1", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_ne_4d_broadcast():
    verify_model("ne_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sqr_2d():
    verify_model("sqr_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_pad1_1():
    verify_model("conv3x3_pad1-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_bilinear_upsample_aligned_constant():
    verify_model("bilinear_upsample_aligned_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_log2_2d():
    verify_model("log2_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_slice():
    verify_model("slice", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv2x2():
    verify_model("deconv2x2", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_all_reduce_spatial():
    verify_model("all_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sqrt_4d():
    verify_model("sqrt_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv7x7_stride4x4():
    verify_model("conv7x7_stride4x4", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_ge_4d():
    verify_model("ge_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_any_reduce_channel():
    verify_model("any_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_and_2d():
    verify_model("and_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_add_4d_broadcast():
    verify_model("add_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_copy_2d():
    verify_model("copy_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_ceil_2d():
    verify_model("ceil_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_linear_squeeze():
    verify_model("linear_squeeze", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sub_2d():
    verify_model("sub_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_valid():
    verify_model("deconv3x3_valid", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_pow_4d():
    verify_model("pow_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pad_1_1_constant():
    verify_model("pad_1-1_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_debox3x3():
    verify_model("debox3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv1x1():
    verify_model("conv1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_exp_4d():
    verify_model("exp_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3_ignore_border():
    verify_model("avg_pool3x3_ignore-border", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_pad0_0():
    verify_model("deconv3x3_pad0-0", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_pow_4d_broadcast():
    verify_model("pow_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_abs_4d():
    verify_model("abs_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sin_4d():
    verify_model("sin_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_select_2d_true():
    verify_model("select_2d_true", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_relu_2d_standalone():
    verify_model("relu_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_reshape_squeeze():
    verify_model("reshape_squeeze", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sub_4d_constant():
    verify_model("sub_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_linear():
    verify_model("linear", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pow_2d():
    verify_model("pow_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_debox3x3_pad0_1():
    verify_model("debox3x3_pad0-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_floor_4d():
    verify_model("floor_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_nobias():
    verify_model("deconv3x3_nobias", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_batch_norm():
    verify_model("batch_norm", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_stride2x2():
    verify_model("deconv3x3_stride2x2", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_debox2x2():
    verify_model("debox2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pad_0_1_replicate():
    verify_model("pad_0-1_replicate", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_mul_4d_broadcast():
    verify_model("mul_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_debox3x3_pad0_0():
    verify_model("debox3x3_pad0-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_argmin_reduce_channel():
    verify_model("argmin_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_copy_4d():
    verify_model("copy_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_not_2d():
    verify_model("not_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sigmoid_4d_standalone():
    verify_model("sigmoid_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_exp_2d():
    verify_model("exp_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_lt_4d():
    verify_model("lt_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv4x4():
    verify_model("conv4x4", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3():
    verify_model("avg_pool3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_avg_pool3x3_pad0_0():
    verify_model("avg_pool3x3_pad0-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv3x3_pad0_1():
    verify_model("conv3x3_pad0-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pad_0_1_constant():
    verify_model("pad_0-1_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv4x4():
    verify_model("deconv4x4", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_neg_2d():
    verify_model("neg_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_bilinear_upsample_asymmetric_replicate():
    verify_model("bilinear_upsample_asymmetric_replicate", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv5x5_stride3x3():
    verify_model("conv5x5_stride3x3", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_relu_4d_standalone():
    verify_model("relu_4d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_max_pool1x1():
    verify_model("max_pool1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv5x5_pad2_2():
    verify_model("deconv5x5_pad2-2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_tile_batch():
    verify_model("tile_batch", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_eq_4d_constant():
    verify_model("eq_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_lt_4d_broadcast():
    verify_model("lt_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv1x1():
    verify_model("deconv1x1", rtol=1e-5, atol=2e-3)


@tvm.testing.parametrize_targets
def test_cts_sign_2d():
    verify_model("sign_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_select_2d_false():
    verify_model("select_2d_false", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_div_4d_constant():
    verify_model("div_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pow_4d_constant():
    verify_model("pow_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_round_4d():
    verify_model("round_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_debox3x3_stride1x1():
    verify_model("debox3x3_stride1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv5x5_stride3x3():
    verify_model("deconv5x5_stride3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sub_4d_broadcast():
    verify_model("sub_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_any_reduce_spatial():
    verify_model("any_reduce_spatial", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_gt_4d_constant():
    verify_model("gt_4d_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv6x6():
    verify_model("conv6x6", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_le_4d():
    verify_model("le_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_gt_4d():
    verify_model("gt_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv4x4_stride2x2():
    verify_model("deconv4x4_stride2x2", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_le_4d_broadcast():
    verify_model("le_4d_broadcast", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_tanh_2d_standalone():
    verify_model("tanh_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box3x3():
    verify_model("box3x3", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_select_4d_false():
    verify_model("select_4d_false", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_tanh():
    verify_model("tanh", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_sin_2d():
    verify_model("sin_2d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box3x3_pad0_0():
    verify_model("box3x3_pad0-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box1x1():
    verify_model("box1x1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_box3x3_pad1_1():
    verify_model("box3x3_pad1-1", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_conv5x5_pad2_2():
    verify_model("conv5x5_pad2-2", rtol=1e-5, atol=1e-2)


@tvm.testing.parametrize_targets
def test_cts_max_pool3x3_pad0_0():
    verify_model("max_pool3x3_pad0-0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_softmax_2d_standalone():
    verify_model("softmax_2d_standalone", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_groups0():
    verify_model("deconv3x3_groups0", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_deconv3x3_pad0_1():
    verify_model("deconv3x3_pad0-1", rtol=1e-5, atol=5e-3)


@tvm.testing.parametrize_targets
def test_cts_sigmoid():
    verify_model("sigmoid", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_argmax_reduce_channel():
    verify_model("argmax_reduce_channel", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pad_1_1_replicate():
    verify_model("pad_1-1_replicate", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_pad_1_0_constant():
    verify_model("pad_1-0_constant", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_unsqueeze():
    verify_model("unsqueeze", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_neg_4d():
    verify_model("neg_4d", rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_cts_add_4d():
    verify_model("add_4d", rtol=1e-5, atol=1e-5)
