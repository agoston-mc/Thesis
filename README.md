# NNEF to TVM converter

Converts an NNEF model into a TVM relay module, ready for compiling and deployment.

Using tvm's [python API](https://tvm.apache.org/docs/reference/api/python/index.html "TVM docs")


## Usage

```{python}
from NNEFConverter import from_nnef 
model, params = from_nnef('path/to/nnef/dir')
``` 
then compile with a TVM executor (autotvm, relay executor, relay build module) 
