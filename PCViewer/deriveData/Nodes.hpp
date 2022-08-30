#pragma once
#include <vector>
#include <array>
#include <stdexcept>
#include <string_view>
#include <list>
#include <map>
#include <memory>
#include <cmath>
#include <iostream>
#include "NodeBase.hpp"
#include "MemoryView.hpp"

namespace deriveData{
// ------------------------------------------------------------------------------------------
// nodes
// ------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------
// special nodes
// ------------------------------------------------------------------------------------------

class InputNode: public Node{
public:
    InputNode(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {}): Node(std::move(inputTypes), std::move(inputNames), std::move(outputTypes), std::move(outputNames), header, mt){};
};

class OutputNode: public Node{
public:
    OutputNode(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {}): Node(std::move(inputTypes), std::move(inputNames), std::move(outputTypes), std::move(outputNames), header, mt){};
};

class DatasetInputNode: public InputNode, public Creatable<DatasetInputNode>{
public:
    std::string_view datasetId;

    DatasetInputNode(std::string_view datasetID = {""}):
        InputNode(createFilledVec<FloatType, Type>(0), {}, createFilledVec<FloatType, Type>(0),{}, "", ""), datasetId(datasetID)
    {
        name = "Dataset Input";
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // Nothing to do. The data is simply forwarded via a view
    };
};

class DataCreationNode: public InputNode{
public:
    DataCreationNode(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {}): InputNode(std::move(inputTypes), std::move(inputNames), std::move(outputTypes), std::move(outputNames), header, mt){};
};

class ZeroVectorNode: public DataCreationNode, public Creatable<ZeroVectorNode>{
public:
    ZeroVectorNode(): DataCreationNode(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "Zero Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].dimensionSizes.empty());            // check for single value constant in input
        assert(input[0].cols.size() && output[0].cols.size());  // check for columns
        assert(output[0].size() == input[0].cols[0][0]);    // enough memory has to be allocated before this call is made..
        for(int i: irange(output[0].size()))
            output[0].cols[0][i] = 0;
    };
};

class OneVectorNode: public DataCreationNode, public Creatable<OneVectorNode>{
public:
    OneVectorNode(): DataCreationNode(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "One Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].dimensionSizes.empty());            // check for single value constant in input
        assert(input[0].cols.size() && output[0].cols.size());  // check for columns
        assert(output[0].size() == input[0].cols[0][0]);    // enough memory has to be allocated before this call is made..
        for(int i: irange(output[0].size()))
            output[0].cols[0][i] = 1;
    };
};

class RandomVectorNode: public DataCreationNode, public Creatable<RandomVectorNode>{
public:
    RandomVectorNode(): DataCreationNode(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "Random Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].dimensionSizes.empty());            // check for single value constant in input
        assert(input[0].cols.size() && output[0].cols.size());  // check for columns
        assert(output[0].size() == input[0].cols[0][0]);    // enough memory has to be allocated before this call is made..
        for(int i: irange(output[0].size()))
            output[0].cols[0][i] = double(rand()) / RAND_MAX;
    };
};

class PrintVectorNode: public OutputNode, public Creatable<PrintVectorNode>{
public:
    PrintVectorNode(): OutputNode(createFilledVec<FloatType, Type>(1), {""}, createFilledVec<FloatType, Type>(0), {}, "Print Vector"){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // prints at max the first 50 and last 50 items of a vector
        std::cout << "[ ";
        if(input[0].columnSize() < 100){
            for(int i: irange(input[0].columnSize())){
                if(input[0].cols.size() > 1)
                    std::cout << "(";
                for(int j: irange(input[0].cols))
                    std::cout << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    std::cout << "), ";
                if((i + 1) % 20 == 0)
                    std::cout << std::endl;
            }
        }
        else{
            for(int i: irange(50)){
                 if(input[0].cols.size() > 1)
                    std::cout << "(";
                for(int j: irange(input[0].cols))
                    std::cout << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    std::cout << "), ";
                if((i + 1) % 20 == 0)
                    std::cout << std::endl;
            }
            std::cout << std::endl << "  ...  " << std::endl;
            for(int i: irange(input[0].columnSize() - 50, input[0].columnSize())){
                if(input[0].cols.size() > 1)
                    std::cout << "(";
                for(int j: irange(input[0].cols))
                    std::cout << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    std::cout << "), ";
                if((i + 1) % 20 == 0)
                    std::cout << std::endl;
            }
        }
        std::cout << "] (size: " << input[0].size() << ", column size: " << input[0].columnSize() << ")" << std::endl;
    };
};

class VariableInput{  // simple class to indicate variable input lengths
public:
    int minNodes;
    int maxNodes;
    
    VariableInput(int minInputs = 0, int maxInputs = std::numeric_limits<int>::max()): minNodes(minInputs), maxNodes(maxInputs){}
};

class DatasetOutputNode: public OutputNode, public VariableInput, public Creatable<DatasetOutputNode>{
public:
    std::string_view datasetId;

    DatasetOutputNode(std::string_view datasetID = {""}):
        OutputNode(createFilledVec<FloatType, Type>(0), {}, createFilledVec<FloatType, Type>(0),{}, "", ""), datasetId(datasetID)
    {
        name = "Dataset Output";
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // Nothing to do. The data has to be moved outside of the node as we need teh vector storing the data to move the data into the dataset
    };
};

class DerivationNode: public Node, public VariableInput, public Creatable<DerivationNode>{
public:
    DerivationNode(): Node(createFilledVec<FloatType, Type>(0), {}, createFilledVec<FloatType, Type>(1), {""}, "Derivation"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO: implement
    };
};

// ------------------------------------------------------------------------------------------
// unary nodes
// ------------------------------------------------------------------------------------------

template<class T>
class UnaryNode: public Node{
public:
    UnaryNode(std::string_view header, std::string_view middle):
        Node(createFilledVec<T, Type>(1), {std::string()}, createFilledVec<T, Type>(1),{std::string()}, header, middle){};
};

class InverseNode: public UnaryNode<FloatType>, public Creatable<InverseNode>{
public:
    InverseNode(): UnaryNode("", "1/"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{     
        assert(input[0].equalDataLayout(output[0]));
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = 1. / input[0].cols[0][i];
    }
};

class NegateNode: public UnaryNode<FloatType>, public Creatable<NegateNode>{
public:
    NegateNode(): UnaryNode("", "*-1"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{        
        assert(input[0].equalDataLayout(output[0]));
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = -input[0].cols[0][i];
    }
};

class NormalizationNode: public UnaryNode<FloatType>, public Creatable<NormalizationNode>{
public:
    enum class NormalizationType{
        ZeroOne,
        MinusOneOne
    };
    NormalizationType normalizationType{NormalizationType::ZeroOne};

    NormalizationNode(): UnaryNode("", "norm"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        auto [mi, ma] = std::minmax(input[0].cols[0].begin(), input[0].cols[0].end());
        float min = *mi;
        float max = *ma;
        if(min == max){
            max += 1;
            max *= 1.1;
        }
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = (input[0].cols[0][i] - min) / (max - min);
    }
};

class AbsoluteValueNode: public UnaryNode<FloatType>, public Creatable<AbsoluteValueNode>{
public:
    AbsoluteValueNode(): UnaryNode("", "abs"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].equalDataLayout(output[0]));
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = std::abs(input[0].cols[0][i]);
    }
};

class SquareNode: public UnaryNode<FloatType>, public Creatable<SquareNode>{
public:
    SquareNode(): UnaryNode("", "square"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].equalDataLayout(output[0]));
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = input[0].cols[0][i] * input[0].cols[0][i];
    }
};

class ExponentialNode: public UnaryNode<FloatType>, public Creatable<ExponentialNode>{
public:
    ExponentialNode(): UnaryNode("", "exp"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].equalDataLayout(output[0]));
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = std::exp(input[0].cols[0][i]);
    }
};

class LogarithmNode: public UnaryNode<FloatType>, public Creatable<LogarithmNode>{
public:
    LogarithmNode(): UnaryNode("", "log"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].equalDataLayout(output[0]));
        for(size_t i: irange(input[0].cols[0].size()))
            output[0].cols[0][i] = std::log(input[0].cols[0][i]);
    }
};

class UnaryVec2Node: public UnaryNode<Vec2Type>{
public:
    UnaryVec2Node(std::string_view header = "", std::string_view body = ""): UnaryNode(header, body){};
};

class CreateVec2Node: public UnaryVec2Node, public Creatable<CreateVec2Node>{
public:
    CreateVec2Node(): UnaryVec2Node(){
        inputTypes = createFilledVec<FloatType, Type>(2);
        inputNames = {"x", "y"};
        outputNames = {"vec2"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class SplitVec2: public UnaryVec2Node, public Creatable<SplitVec2>{
public:
    SplitVec2(): UnaryVec2Node(){
        inputNames = {"vec2"};
        outputTypes = createFilledVec<FloatType, Type>(2);
        outputNames = {"x", "y"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class Vec2Norm: public UnaryVec2Node, public Creatable<Vec2Norm>{
public:
    Vec2Norm(): UnaryVec2Node("", "len"){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class UnaryVec3Node: public UnaryNode<Vec3Type>{
public:
    UnaryVec3Node(std::string_view header = "", std::string_view body = ""): UnaryNode(header, body){};
};

class CreateVec3Node: public UnaryVec3Node, public Creatable<CreateVec3Node>{
public:
    CreateVec3Node(): UnaryVec3Node(){
        inputTypes = createFilledVec<FloatType, Type>(3);
        inputNames = {"x", "y", "z"};
        outputNames = {"vec3"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class SplitVec3: public UnaryVec3Node, public Creatable<SplitVec3>{
public:
    SplitVec3(): UnaryVec3Node(){
        inputNames = {"vec3"};
        outputTypes = createFilledVec<FloatType, Type>(3);
        outputNames = {"x", "y", "z"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class Vec3Norm: public UnaryVec3Node, public Creatable<Vec3Norm>{
public:
    Vec3Norm(): UnaryVec3Node("", "len"){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class UnaryVec4Node: public UnaryNode<Vec4Type>{
public:
    UnaryVec4Node(std::string_view header = "", std::string_view body = ""): UnaryNode(header, body){};
};

class CreateVec4Node: public UnaryVec4Node, public Creatable<CreateVec4Node>{
public:
    CreateVec4Node(): UnaryVec4Node(){
        inputTypes = createFilledVec<FloatType, Type>(4);
        inputNames = {"x", "y", "z", "w"};
        outputNames = {"vec4"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class SplitVec4: public UnaryVec4Node, public Creatable<SplitVec4>{
public:
    SplitVec4(): UnaryVec4Node(){
        inputNames = {"vec4"};
        outputTypes = createFilledVec<FloatType, Type>(4);
        outputNames = {"x", "y", "z", "w"};
    };

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

class Vec4Norm: public UnaryVec4Node, public Creatable<Vec4Norm>{
public:
    Vec4Norm(): UnaryVec4Node("", "len"){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // TODO implement
    }
};

// ------------------------------------------------------------------------------------------
// binary nodes
// ------------------------------------------------------------------------------------------

template<class T>
class BinaryNode: public Node{
public:
    BinaryNode(std::string_view header = "", std::string_view body = ""):
        Node(createFilledVec<T, Type>(2), {std::string(), std::string()}, createFilledVec<T, Type>(1), {std::string()}, header, body){};
};

class PlusNode: public BinaryNode<FloatType>, public Creatable<PlusNode>{
public:
    PlusNode(): BinaryNode("", "+"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].size() == output[0].size() || input[1].size() == output[0].size()); // only one needs to have the same size
        if(input[0].equalDataLayout(input[1]) && input[0].equalDataLayout(output[0])){ // fast addition possible
            for(long i: irange(output[0].cols[0].size())){
                output[0].cols[0][i] = input[0].cols[0][i] + input[1].cols[0][i];
            }
        }
        else{   // less efficient but more general
            for(long i: irange(output[0].size())){
                output[0](i, 0) = input[0](i, 0) + input[1](i, 0);
            }
        }
    }
};

class MinusNode: public BinaryNode<FloatType>, public Creatable<MinusNode>{
public:
    MinusNode(): BinaryNode("", "-"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].size() == output[0].size() || input[1].size() == output[0].size()); // only one needs to have the same size
        if(input[0].equalDataLayout(input[1]) && input[0].equalDataLayout(output[0])){ // fast addition possible
            for(long i: irange(output[0].cols[0].size())){
                output[0].cols[0][i] = input[0].cols[0][i] - input[1].cols[0][i];
            }
        }
        else{   // less efficient but more general
            for(long i: irange(output[0].size())){
                output[0](i, 0) = input[0](i, 0) - input[1](i, 0);
            }
        }
    }
};

class MultiplicationNode: public BinaryNode<FloatType>, public Creatable<MultiplicationNode>{
public:
    MultiplicationNode(): BinaryNode("", "*"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].size() == output[0].size() || input[1].size() == output[0].size()); // only one needs to have the same size
        if(input[0].equalDataLayout(input[1]) && input[0].equalDataLayout(output[0])){ // fast addition possible
            for(long i: irange(output[0].cols[0].size())){
                output[0].cols[0][i] = input[0].cols[0][i] * input[1].cols[0][i];
            }
        }
        else{   // less efficient but more general
            for(long i: irange(output[0].size())){
                output[0](i, 0) = input[0](i, 0) * input[1](i, 0);
            }
        }
    }
};

class DivisionNode: public BinaryNode<FloatType>, public Creatable<DivisionNode>{
public:
    DivisionNode(): BinaryNode("", "/"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        assert(input[0].size() == output[0].size() || input[1].size() == output[0].size()); // only one needs to have the same size
        if(input[0].equalDataLayout(input[1]) && input[0].equalDataLayout(output[0])){ // fast addition possible
            for(long i: irange(output[0].cols[0].size())){
                output[0].cols[0][i] = input[0].cols[0][i] / input[1].cols[0][i];
            }
        }
        else{   // less efficient but more general
            for(long i: irange(output[0].size())){
                output[0](i, 0) = input[0](i, 0) / input[1](i, 0);
            }
        }
    }
};
}