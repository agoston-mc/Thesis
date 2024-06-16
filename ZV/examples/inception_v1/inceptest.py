import logging
import random
import os

import nnef
import nnef_tools.execute as ex
import numpy
import numpy as np
import tvm
import tvm.relay as relay

import NNEFConverter as nc

os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"

params = {'data_0': np.array([[[[random.randint(1, 255) for _ in range(224)] for _ in range(224)],
                               [[random.randint(1, 255) for _ in range(224)] for _ in range(224)],
                               [[random.randint(1, 255) for _ in range(224)] for _ in range(224)]]], dtype='float32')}

import PIL.Image

img = PIL.Image.open('elephant.jpg')

im = img.resize((224, 224))
im = im.convert('RGB')

# im.show()

x = numpy.array(im, dtype='float32').transpose(2, 0, 1)
x = np.expand_dims(x, axis=0)

params['data_0'] = x

# params['data_0'] = np.full([1, 3, 224, 224], 154.0, dtype='float32')

print(params['data_0'].shape)

nnef_exe = ex.NNEFExecutor('inception_v1.nnef', None, None)

neo = nnef_exe(params)

# print(neo[0]['concat'][0][:10], end='\n\n')


mod, tparams = nc.from_nnef('inception_v1.nnef')

# print(mod)

target = 'llvm'  #tvm.target.cuda(arch='sm_89')
with tvm.transform.PassContext(opt_level=5):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, tparams
    )
    # lib = relay.build(mod, target, params=tparams)

executor = executor.evaluate()
tvm_output = executor(tvm.nd.array(params['data_0'])).numpy()

print('ne  conf: ', end='')
print(max(*neo[0]['prob_1'][0]))
print('tvm conf: ', end='')
print(max(*tvm_output[0]))
# print('tvm lib m conf: ', end='')
# print(max(*extor.get_output(0).numpy()[0]))
print('ne  guess: ', end='')
print(max(enumerate(neo[0]['prob_1'][0]), key=lambda y: y[1])[0])
print('tvm guess: ', end='')
print(max(enumerate(tvm_output[0]), key=lambda y: y[1])[0])
# print('tvm lib m guess: ', end='')
# print(max(enumerate(extor.get_output(0).numpy()[0]), key=lambda y: y[1])[0])

from tvm.contrib import graph_executor

target = tvm.target.Target("llvm")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("data_0", x)
module.run()
tvm_output = module.get_output(0).numpy()
print('tvm relay execute')
print(tvm_output[0][:10])