- Feature Name: `Relay NNEF frontend`
- Start Date: 2024-04-11
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Add a Neural Network Exchange Format frontend to TVM.

# Motivation
[motivation]: #motivation

Why are we doing this? What use cases does it support? What is the expected outcome?


PaddlePaddle, an independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 2.3 million developers. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.

Currently, PaddlePaddle has built a prosperous technological ecology, there are more than 500 models developed by official organization or outside developers, covering CV/NLP/OCR/Speech, refer to the following links for more details,

- [PaddlePaddle/models](https://github.com/PaddlePaddle/models)
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
- [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)

As of version 2.0, PaddlePaddle supports imperative programming like PyTorch. Furthermore, a mechanism of `Dynamic to Static` is provided to export a PaddlePaddle model to graph representation, which is more friendly for deployment. The following example code shows how to export a PaddlePaddle model,

```
import paddle
import paddlehub
model = hub.Module(name="resnet50_vd_imagenet_ssld")
input_spec = paddle.static.InputSpec(
    [1, 3, 224, 224], "float32", "image")
paddle.jit.save(model, "model/infer", input_spec=[input_spec])
```

PaddlePaddle's deployment is supported by Paddle Inference/Paddle Lite/OpenVINO/Tengine/Adlik now. We noticed that there are lots of developers converting models to ONNX format for the compatibility with TVM, but only a limited number of models are convertible due to lack of ONNX operators.  
Based on this background, we proposed this RFC to add a PaddlePaddle frontend for TVM, improving usability for PaddlePaddle users and enhancing the compatibility between PaddlePaddle and TVM.



# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We are going to add an NNEF support, for that we can use either a NNEF model directory, or an `nnef.Graph` object 
already loaded into memory.
The conversion is done via the new frontend function
```python
relay.frontend.from_nnef(model, freeze_vars=False)
```
  - model: either a string, or PathLike to an NNEF model directory, or an `nnef.Graph` object.
  - freeze_vars: optional bool, which sets whether the parameters should be considered variables or constants for optimisation

Example usages (assume we have a directory `inception_v1.nnef` with a complete NNEF Inception graph)
```python
import nnef
from tvm import relay

model_path = 'path/to/model/inception_v1.nnef'
# If modification is needed the graph can be imported with `nnef.load_graph` 
graph = nnef.load_graph(model_path)

mod, params = relay.frontend.from_nnef(graph)
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

As this RFC only adds a new frontend, no other features should be affected. 

The process of importing a NNEF model consists of:

- Loading an NNEF model into memory, if a model path was provided `nnef.load_graph` is used to get an `nnef.Graph` object.
This is supported so the model can be used, or modified before conversion with methods provided for NNEF.
- Converting the operations of the Graph, setting inputs, and reading parameters one by one.


# Drawbacks
[drawbacks]: #drawbacks

Potential increase in time-cost of unit tests.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The frontend of NNEF is similar to that of ONNX, PyTorch, and TensorFlow. 
Currently, as NNEF does not have an official compiler, the workflow with NNEF graphs is rather complicated, 
possible through conversion to other models. 

# Prior art
[prior-art]: #prior-art

These are the projects currently supporting with NNEF:

- https://github.com/sonos/tract
- https://github.com/fragata-ai/arhat-nnef
- https://rocm.docs.amd.com/projects/MIVisionX/en/latest/model_compiler/README.html



# Unresolved questions
[unresolved-questions]: #unresolved-questions

`-`

# Future possibilities
[future-possibilities]: #future-possibilities

We will be working on supporting the next generation of NNEF as well, while some NNEF methods are not supported by 
this frontend, they were deemed not necessary as they are not widely used, if needed, they can be implemented as well.
