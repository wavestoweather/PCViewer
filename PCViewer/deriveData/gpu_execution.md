# Implementation doumentation and theory behind gpu data derivation

## Introdution
Executing the compute graph of the derivation nodes on the gpu in merged kernel style requires carful treatment of dimension changes. Eg. it is not trivial to process a summation of a 2D field quantit with a 3D field, where the 2D field is defined as the bottom slice of the 3D domain.

## Pipeline overview
1. Execution graph parsing: In the first step the execution graph is traversed from output nodes backdwards. this automically prunes all nodes with no effect. Further this ensures, that the output nodes will be the last execution nodes in the pipeline, making it very likely that they will survive. The discrete steps are as follows:
    - Go over all output nodes and start iterative node execution over all input nodes
    - Keep track of input and output dimensions, as well as in place opportunities and allocate new data vectors or reuse old ones. Do not instantly create new data vectors, only create new data vectors when inflatoin happens.
    - According to a newly created buffer, a already created buffer and a constant set the `std::variant<address, const_address, float>` to the corresponding value of either the buffer address of the GPU-buffer or teh value of the constant.
    - For each input noode call the "record_instruction" method, which puts its op_code into the operation_code_stream for further processing
2. This first list is then handed over to an optimizer, which tries to identify not needed load/store operations and thus effectively merges kernels. This is done by keeping track of the storage spaces the data was stored, its data layout and when one operation stores something inside the buffer, and later that buffer is read without any interventions from other operations on that buffer of other buffers for that matter, the `pipeline_barrier` command is removed and the `storage` vector components are rotated to have the data in the correct place for later usage (uses special `rotate` operation)
3. The so created assembly like list of instructions is then passed on the the `deriveData::create_gpu_pipelines()` function. This function takes the intermediate representation, creates compute shader with all the instructions baked in as well as possible (Eg. if operations can be done in place no pipeline synchronization is needed), and returns an array of execution informations which can then be recorded into a command buffer and submitted for execution.

## Dimensionality discrepancy
In this implementation dimension discrepancies will always be handled the following way:

The resulting dimensionality of an operation (If normal mathematical operation is given) is the cartesian product of the unique dimensions (or over the dimension sets). As this means, that at least one input will be inflated the data has to be reloaded for the inflated data (If no data was loaded already, simply load the data inflated).

After the node operation has been carried out, the result is first kept in memory. Should the next operation require an inflation as well, the data has to be stored first by the new operation and has to be reloaded inflated.

Note: For optimal execution times, data which has never to be inflated should be kept in memory. This means, that all inflation operations should have been performed until they are merged with the largest uninflated data block. This reduces store and load operations where not necessary. A possible immplementation is to store a dependency graph including in between storage information. On execution execute first of all the pipelines requiring inflation and in the end only run kernels which do not require inflation. My current understaning however is, that due to the way the traversal works, these things are already accounted for.

## deriveData::create_gpu_pipelines() description
The instruction list given to the compilation unit already has information about additionally required storage buffers baked in. Addresses which can reside in memory do have a "l" prefix (meaning local). Addresses which are storage buffers have a "g" prefix (as in global) and require a loading/storing operation from/to memory. The data access to retrieve/store the data from/to memory is decided by the computation node. Constants (single non changing values) are marked by a "c"-prefix and are directly put into the shader (Constants are already stored by their numeric value in the instruction code).

There is no need for logic deciding if the operation has to be stored in global or local memory, instead it can be determmined by the constantness of the addresses:
### inputs
- l: If an address is local, the data can be loaded directly from the `float storage[]` array.
- g: A global address means that the data first has to be fetched into the `float storage[]` array before going further
- c: The constant should not be put into the `float storage[]` array at all, but should be directly infused into the operation instead of the storage lookup

### outputs
For the outputs things are more complex. Basically always first of all store the data in the local `float storage[]` array. If Storing to global memroy is required a store operation will be invoked.

Special care has to be taken for nodes such as reduction nodes, as they require atomic operations into a global storage buffer
