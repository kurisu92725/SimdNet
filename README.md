# SimdNet
Simplified Directional Message Passing Neural Network for Prediction of Protein-Ligand Binding Affinity

http://dx.doi.org/10.2139/ssrn.4376917

##Code Environment
- **key packages**
    ```python
    PyG    https://github.com/pyg-team/pytorch_geometric
    DGL    https://github.com/dmlc/dgl
    DIG    https://github.com/divelab/DIG
    RDKit  http://www.rdkit.org/
    ```
  
## Usage
**Training**


After setting the file directory for the dataset, adjust the parameters as required.

In a run-time environment, in a project directory, train the model using the following commands:
  ```python
  python simd_train.py
  ```
Note that because the pre-processing training set is large, you can download the original data set and process it yourself. The web address of the original data set is: http://www.pdbbind.org.cn/

**Testing**

Test with the following code:
  ```python
  python prediction.py
  ```
Pre-processed PDBBind-v2016-coreset is already included in the project. For the convenience of testing, the network parameters of the example are also given.


## Problems
One possible problem is the call of Swish activation function, which you can specify. And for calls in the `dig` package, you can create a corresponding file in the `pytorch_geometric` package.

  


