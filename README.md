# NNEF to TVM converter

Converts an NNEF model into a TVM relay module, ready for compiling and deployment.
Using the python API of tvm


## Usage

```{python}
from NNEFConverter import from_nnef 
model, params = from_nnef('path_to_nnef_dir')
``` 
then compile with a TVM executor (autotvm, relay executor) 

