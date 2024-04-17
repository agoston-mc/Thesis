# Add a Neural Network Exchange Format frontend to TVM

Hi, we are developers from aiMotive, member company of the Khronos Group (https://www.khronos.org), working on creating an NNEF frontend for TVM relay. 

# Motivation
[motivation]: #motivation


NNEF is an open, standardized format for neural network exchange developed by the Khronos Group since 2018 (https://www.khronos.org/nnef). It is aimed at deploying trained neural networks from deep learning frameworks to proprietary inference engines of neural network hardware vendors. Such inference engines often require an offline compilation step for running models more efficiently, hence hardware vendors are are looing into open source compiler stacks to be leveraged. On one hand, hardware vendors may integrate their hardware as a backend into TVM, while at the same time integrating NNEF as a frontend would allow vendors to use TVM as an end-to-end compilation tool starting from a standardized format.

The Khronos Group also maintains a set of tools for handling NNEF models. Since NNEF is mainly a textual format, these include a parser (with C++ and Python interfaces), and conversion tools from other formats. NNEF supports conversion from models of various deep learning frameworks, including Caffe, TensorFlow (also Lite) and all those that support ONNX, such as PyTorch. Creating NNEF models is also possible manually by directly writing the model text file(s) (since NNEF is similar to a scripting language). Manually written models may even be executed or trained in deep learning frameworks (currently support for PyTorch exists).



# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The frontend of NNEF is similar to that of ONNX, PyTorch, and TensorFlow, adding it would increase the number of model formats that TVM can process.


# Current progress

The current development supports the conversion of the majority of the operations (97), with the remaining operations considered low priority, and not planned to be supported for now.

Test cases have been written to cover the finished operators, and the converter has also been tested on popular complete models with success. 



# Future possibilities
[future-possibilities]: #future-possibilities

The Khronos Groups is actively working on the next major update to the NNEF format, whose main purpose is to increase model coverage by adding support for dynamic models and custom operators. In the latter case, more involved compilation of models carries even more potential, so we plan to add support for the next generation as well.

Support for some NNEF operators would only be possible through more complex mapping to a sequence of TVM operators, and the less widely used ones were not the focus of this initial release. We may add support to such operators in the future if required.
