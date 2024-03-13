import os
import nnef
from nnef_ops import _get_converter_map
from nnef_ops import *

from tvm.ir import IRModule
from tvm.relay import analysis, function
from tvm.relay.frontend.common import new_var, fold_constant, set_span


# Converter class
class NNEF_Converter:

    def __init__(self, freeze_vars):
        self._nodes = {}
        self._consts = {}
        self._inputs = {}
        self._num_inputs = 0
        self._params = {}
        self._num_params = 0
        self._freeze_vars = freeze_vars

    def from_nnef(self, graph):
        self._parse_inputs(graph)
        self._construct_nodes(graph)

        outputs = [self._nodes[n] for n in graph.outputs]
        outputs = outputs[0] if len(outputs) == 1 else tmv_expr.Tuple(outputs)

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
            # if inp in self._params:
            #     self._num_params += 1
            #     self._nodes[inp] = new_var(inp, shape=self._params[inp].shape, dtype=self._params[inp].dtype)
            # elif inp in self._nodes:
            #     continue
            # else:
            self._num_inputs += 1
            i_tens = graph.tensors[inp]
            self._nodes[inp] = new_var(inp, shape=i_tens.shape, dtype=get_type(i_tens.dtype))
            self._inputs[inp] = self._nodes[inp]

    def _construct_nodes(self, graph):
        for op in graph.operations:
            if op.name == 'external':
                continue

            if op.name == 'variable':
                tensor = graph.tensors[op.outputs['output']]
                tens_data = tensor.data
                if self._freeze_vars:
                    self._consts[tensor.name] = tmv_expr.const(tens_data)
                    self._nodes[tensor.name] = self._consts[tensor.name]
                else:
                    self._nodes[tensor.name] = new_var(tensor.name, shape=op.attribs['shape'],
                                                       dtype=get_type(tensor.dtype))
                    self._params[tensor.name] = tens_data

            elif op.name == 'constant':
                self._set_const(op)

            else:
                self._set_literal_inputs(op)
                self._set_parameter_span(op, op.name)
                inputs = []
                for ink, inv in op.inputs.items():
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

                converted = self._get_relay_op_call(op.name, inputs, op.attribs)

                if not isinstance(converted, tmv_expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(converted)

                if outputs_num == 1:
                    if not isinstance(converted, tmv_expr.TupleWrapper):
                        converted = fold_constant(converted)
                    else:
                        converted = fold_constant(converted.astuple())
                else:
                    converted = tmv_expr.TupleWrapper(fold_constant(converted.astuple()), len(converted))

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

    def _set_const(self, node):
        name = node.outputs['output']
        data = node.attribs['value']
        shape = node.attribs['shape']
        if len(data) == 1:
            data = np.full(shape, data, dtype=get_type(node.dtype))
        else:
            data = np.array(data, dtype=get_type(node.dtype))
        self._consts[name] = tmv_expr.const(data)
        self._nodes[name] = self._consts[name]

    def _set_literal_inputs(self, node):
        for k, v in node.inputs.items():
            if isinstance(v, list):
                for ve in v:
                    if ve not in self._nodes.keys():
                        self._nodes[f'{node.name}_{k}'] = tmv_expr.const(np.array(ve, dtype=get_type(node.dtype)))

            else:
                if v not in self._nodes.keys():
                    if node.dtype:
                        dtype = get_type(node.dtype)
                    else:
                        dtype = infer_type(v)
                    self._nodes[f'{node.name}_{k}'] = tmv_expr.const(np.array(v, dtype=dtype))

    def _set_parameter_span(self, node, node_source_name):
        for k, name in node.inputs.items():
            if isinstance(name, list):
                for n in name:
                    self._set_par_span_helper(node, node_source_name, n)
            else:
                self._set_par_span_helper(node, node_source_name, name)

    def _set_par_span_helper(self, node, node_source_name, name):
        expr = self._nodes.get(name)

        if isinstance(expr, relay.Constant):
            if name not in self._consts:
                name = f'{node.name}_const'
        if isinstance(expr, relay.Var):
            expr_with_span = set_span(expr, make_parameter_span([node_source_name, name]))
            self._nodes[name] = expr_with_span
            if name in self._inputs:
                self._inputs[name] = expr_with_span

        raise TypeError(f'Failed to interpret {name}, while setting the span for {node_source_name}')

    def _get_relay_op_call(self, name, inputs, attrs):
        conv_map = _get_converter_map()
        if name in conv_map:
            call = conv_map[name](*inputs, **attrs)
        else:
            raise NotImplementedError(f'Operator {name} is not implemented.')
        return call


def from_nnef(
        model_path: os.PathLike | str,
        freeze_vars=False
):
    """
    :return: (mod, params) : (tvm.IRModule, dict of str and tvm.nd.NDArray)
    """
    par = NNEF_Converter(freeze_vars)
    model = nnef.load_graph(model_path)
    nnef.infer_shapes(model)
    return par.from_nnef(graph=model)
