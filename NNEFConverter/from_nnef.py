import os
import nnef
from .nnef_ops import _get_converter_map
from .nnef_ops import *

from tvm.ir import IRModule
from tvm.relay import analysis, function
from tvm.relay.frontend.common import new_var, fold_constant, set_span, infer_type


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
        outputs = outputs[0] if len(outputs) == 1 else tvm_expr.Tuple(outputs)

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
                    self._consts[tensor.name] = tvm_expr.const(tens_data)
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

                if not isinstance(converted, tvm_expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(converted)

                if outputs_num == 1:
                    if not isinstance(converted, tvm_expr.TupleWrapper):
                        converted = fold_constant(converted)
                    else:
                        converted = fold_constant(converted.astuple())
                else:
                    converted = tvm_expr.TupleWrapper(fold_constant(converted.astuple()), len(converted))

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
        self._consts[name] = tvm_expr.const(data)
        self._nodes[name] = self._consts[name]

    def _set_literal_inputs(self, node):
        for k, v in node.inputs.items():
            if isinstance(v, list):
                for ve in v:
                    dtype, is_literal = self._infer_type(ve)
                    if is_literal:
                        self._nodes[f'{node.name}_{k}'] = tvm_expr.const(np.array(ve, dtype=get_type(node.dtype)))

            else:
                dtype, is_literal = self._infer_type(v)
                if is_literal:
                    self._nodes[f'{node.name}_{k}'] = tvm_expr.const(np.array(v, dtype=dtype))

    def _set_parameter_span(self, node, node_source_name):
        for field_name, name in node.inputs.items():
            if isinstance(name, list):
                for n in name:
                    self._set_par_span_helper(node, node_source_name, n, field_name)
            else:
                self._set_par_span_helper(node, node_source_name, name, field_name)

    def _set_par_span_helper(self, node, node_source_name, name, field_name):
        _, is_literal = self._infer_type(name)
        if is_literal:
            name = f'{node.name}_{field_name}'

        expr = self._nodes.get(name)

        if isinstance(expr, relay.Constant):
            if name not in self._consts:
                name = f'{node.name}_const'
        expr_with_span = set_span(expr, make_parameter_span([node_source_name, name]))
        self._nodes[name] = expr_with_span
        if name in self._inputs:
            self._inputs[name] = expr_with_span

        # if not isinstance(expr, relay.expr.RelayExpr):
        #     raise TypeError(f'Failed to interpret {name}, while setting the span for {node_source_name}')

    def _get_relay_op_call(self, name, inputs, attrs):
        conv_map = _get_converter_map()
        if name in conv_map:
            call = conv_map[name](*inputs, **attrs)
        else:
            # This error is reached if NNEF is expanded with additional ops
            raise NotImplementedError(f'Operator {name} is not implemented, as {name} has been added after 1.0.5.')
        return call

    def _infer_type(self, val):
        if isinstance(val, bool):
            return 'bool', True
        if isinstance(val, float):
            return 'float32', True
        if isinstance(val, int):
            return 'int32', True
        if isinstance(val, str):
            # the string val's can be names of nodes in some of the cases
            if isinstance(val, nnef.Identifier):
                if val in self._nodes.keys():
                    node = self._nodes[val]
                    if isinstance(node, tvm_expr.Var):
                        return node.type_annotation.dtype, False
                    if isinstance(node, tvm_expr.Constant):
                        return node.data.dtype, False
                    if isinstance(node, tvm_expr.Call):
                        return infer_type(node).checked_type.dtype, False
                raise Exception(f'{val} has not been loaded into the model but it should have been, as a var or call.')
            return 'string', True

        raise TypeError(f'Value \'{val}\' is not a recognized type')


def from_nnef(
        model_path: os.PathLike | str,
        freeze_vars=False
):
    """
    Convert an NNEF model into an equivalent TVM Relay IRModule.


    Parameters
    ----------
    model_path : str or os.PathLike
        path to an NNEF model directory, containing the graph.nnef (and weight files)

    freeze_vars : bool, optional
        If this parameter is true, the nnef variables will be converted to
        constants, and be embedded into the relay model, allowing optimizations
        at compile time.

    Returns
    -------
    mod : tvm.IRModule
        the relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dictionary to be used
    """
    conv_clss = NNEF_Converter(freeze_vars)
    model = nnef.load_graph(model_path)

    # fills in the nnef graph's shape information
    nnef.infer_shapes(model)

    return conv_clss.from_nnef(graph=model)
