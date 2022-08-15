#pragma once
#include <vector>
#include <array>
#include <stdexcept>
#include <string_view>
#include <list>
#include <map>
#include <memory>
#include <initializer_list>
#include <cmath>
#include "../range.hpp"
#include "../Structures.hpp"
#include "../imgui_nodes/imgui_node_editor.h"
#include "../imgui_nodes/utilities/widgets.h"

namespace deriveData{
template<class T, class Base = T>
class Creatable{
public:
    template<typename... Args>
    static std::unique_ptr<T> create(Args&&... args){
        return std::make_unique<T>(args...);
    }
    template<typename... Args>
    static std::unique_ptr<Base> createBase(Args&&... args){
        std::unique_ptr<T> ptr = std::make_unique<T>(args...);
        return std::unique_ptr<Base>(ptr.release());
    }
};

template<class T, class Base = T>
inline std::vector<std::unique_ptr<Base>> createFilledVec(uint32_t size){
    std::vector<std::unique_ptr<Base>> vec(size);
    for(int i: irange(size))
        vec[i] = T::createBase();
    return vec;
}

// ------------------------------------------------------------------------------------------
// types
// ------------------------------------------------------------------------------------------

class Type{
public:
    virtual std::array<float, 4> color() const = 0;
    ax::Widgets::IconType iconType() const{return ax::Widgets::IconType::Circle;};
};

class FloatType: public Type, public Creatable<FloatType, Type>{
public:
    std::array<float, 4> color() const override{return {1,0,0,1};};
};

class ConstantFloatType: public Type, public Creatable<ConstantFloatType, Type>{
    std::array<float, 4> color() const override{return {1, .5, 0, 1};};
};

class VectorType: public Type,  public Creatable<VectorType, Type>{
public:
    std::array<float, 4> color() const override{return {0,1,0,1};};
};

class Vec2Type: public Type,  public Creatable<Vec2Type, Type>{
public:
    std::array<float, 4> color() const override{return {.5, .5, .5, 1};};
};

class Vec3Type: public Type,  public Creatable<Vec3Type, Type>{
public:
    std::array<float, 4> color() const override{return {1,1,1,1};};
};

class Vec4Type: public Type,  public Creatable<Vec4Type, Type>{
public:
    std::array<float, 4> color() const override{return {.1, .1, .1, 1};};
};



// ------------------------------------------------------------------------------------------
// nodes
// ------------------------------------------------------------------------------------------

class Node{
public:
    std::vector<std::unique_ptr<Type>> inputTypes;
    std::vector<std::string> inputNames;
    std::vector<std::unique_ptr<Type>> outputTypes;
    std::vector<std::string> outputNames;

    std::string name;

    Node(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}):
        inputTypes(std::move(inputTypes)),
        inputNames(inputNames),
        outputTypes(std::move(outputTypes)),
        outputNames(outputNames){}


    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const = 0;
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const = 0;

    static void alignInputAndOutput(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output){
        output.resize(input.size());
        for(int i: irange(input))
            output[i].resize(input[i].size());
    }
};

// ------------------------------------------------------------------------------------------
// special nodes
// ------------------------------------------------------------------------------------------

class DatasetInputNode: public Node{
public:
    const std::string_view datasetId;
    const std::list<DataSet>& datasets;

    DatasetInputNode(std::string_view datasetId, const std::list<DataSet>& datasets, const std::vector<Attribute>& attributes):
        datasets(datasets)
    {
        const auto& d = getDataset(datasets, datasetId);
        for(int i: irange(d.data.columns.size())){
            outputTypes.push_back(FloatType::create());
            outputNames.push_back(attributes[i].originalName);
        }
        name = "Input data: " + d.name;
    }

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    };
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    };
};

class DerivationNode: public Node{
public:
    // TODO: implement later...
};

// ------------------------------------------------------------------------------------------
// unary nodes
// ------------------------------------------------------------------------------------------

template<class T>
class UnaryNode: public Node{
public:
    UnaryNode():
        Node(createFilledVec<T, Type>(1), {std::string()}, createFilledVec<T, Type>(1),{std::string()}){};
};

class MultiplicationInverseNode: public UnaryNode<FloatType>{
public:
    MultiplicationInverseNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        Node::alignInputAndOutput(input, output);
        
        for(size_t i: irange(input)){
            for(size_t j: irange(input[i]))
                output[i][j] = 1. / input[i][j];
        }
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        for(size_t i: irange(inout)){
            for(size_t j: irange(inout[i]))
                inout[i][j] = 1. / inout[i][j];
        }
    }
};

class AdditionInverseNode: public UnaryNode<FloatType>{
public:
    AdditionInverseNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        Node::alignInputAndOutput(input, output);
        
        for(size_t i: irange(input)){
            for(size_t j: irange(input[i]))
                output[i][j] = -input[i][j];
        }
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        for(size_t i: irange(inout)){
            for(size_t j: irange(inout[i]))
                inout[i][j] = -inout[i][j];
        }
    }
};

class NormalizationNode: public UnaryNode<FloatType>{
public:
    enum class NormalizationType{
        ZeroOne,
        MinusOneOne
    };
    NormalizationType normalizationType{NormalizationType::ZeroOne};

    NormalizationNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class AbsoluteValueNode: public UnaryNode<FloatType>{
public:
    AbsoluteValueNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        alignInputAndOutput(input, output);

        for(size_t i: irange(input)){
            for(size_t j: irange(input))
                output[i][j] = std::abs(input[i][j]);
        }
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        for(size_t i: irange(inout)){
            for(size_t j: irange(inout[i]))
                inout[i][j] = std::abs(inout[i][j]);
        }
    }
};

class SquareNode: public UnaryNode<FloatType>{
public:
    SquareNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        alignInputAndOutput(input, output);

        for(size_t i: irange(input)){
            for(size_t j: irange(input))
                output[i][j] = input[i][j] * input[i][j];
        }
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        for(size_t i: irange(inout)){
            for(size_t j: irange(inout[i]))
                inout[i][j] = inout[i][j] * inout[i][j];
        }
    }
};

class ExponentialNode: public UnaryNode<FloatType>{
public:
    ExponentialNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        alignInputAndOutput(input, output);

        for(size_t i: irange(input)){
            for(size_t j: irange(input))
                output[i][j] = std::exp(input[i][j]);
        }
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        for(size_t i: irange(inout)){
            for(size_t j: irange(inout[i]))
                inout[i][j] = std::exp(inout[i][j]);
        }
    }
};

class LogarithmNode: public UnaryNode<FloatType>{
public:
    LogarithmNode(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class UnaryVec2Node: public UnaryNode<Vec2Type>{
public:
    UnaryVec2Node(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class CreateVec2Node: public UnaryVec2Node{
public:
    CreateVec2Node(): UnaryVec2Node(){
        inputTypes = createFilledVec<FloatType, Type>(2);
        inputNames = {"", ""};
    }

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class SplitVec2: public UnaryVec2Node{
public:
    SplitVec2(): UnaryVec2Node(){
        outputTypes = createFilledVec<FloatType, Type>(2);
        outputNames = {"", ""};
    }

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class Vec2Norm: public UnaryVec2Node{
public:
    Vec2Norm(): UnaryVec2Node(){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class UnaryVec3Node: public UnaryNode<Vec3Type>{
public:
    UnaryVec3Node(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class CreateVec3Node: public UnaryVec3Node{
public:
    CreateVec3Node(): UnaryVec3Node(){
        inputTypes = createFilledVec<FloatType, Type>(3);
        inputNames = {"", "", ""};
    }

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class SplitVec3: public UnaryVec3Node{
public:
    SplitVec3(): UnaryVec3Node(){
        outputTypes = createFilledVec<FloatType, Type>(3);
        outputNames = {"", "", ""};
    }

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class Vec3Norm: public UnaryVec3Node{
public:
    Vec3Norm(): UnaryVec3Node(){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class UnaryVec4Node: public UnaryNode<Vec4Type>{
public:
    UnaryVec4Node(): UnaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class CreateVec4Node: public UnaryVec4Node{
public:
    CreateVec4Node(): UnaryVec4Node(){
        inputTypes = createFilledVec<FloatType, Type>(4);
        inputNames = {"", "", ""};
    }

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class SplitVec4: public UnaryVec4Node{
public:
    SplitVec4(): UnaryVec4Node(){
        outputTypes = createFilledVec<FloatType, Type>(4);
        outputNames = {"", "", "", ""};
    };

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class Vec4Norm: public UnaryVec4Node{
public:
    Vec4Norm(): UnaryVec4Node(){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

// ------------------------------------------------------------------------------------------
// binary nodes
// ------------------------------------------------------------------------------------------

template<class T>
class BinaryNode: public Node{
public:
    BinaryNode():
        Node(createFilledVec<T, Type>(2), {std::string(), std::string()}, createFilledVec<T, Type>(1), {std::string()}){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class PlusNode: public BinaryNode<FloatType>{
public:
    PlusNode(): BinaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class MinusNode: public BinaryNode<FloatType>{
public:
    MinusNode(): BinaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class MultiplicationNode: public BinaryNode<FloatType>{
public:
    MultiplicationNode(): BinaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};

class DivisionNode: public BinaryNode<FloatType>{
public:
    DivisionNode(): BinaryNode(){};

    virtual void applyOperationCpu(const std::vector<std::vector<float>>& input ,std::vector<std::vector<float>>& output) const override{
        // TODO implement
    }
    virtual void applyOperationInplaceCpu(std::vector<std::vector<float>>& inout) const override{
        // TODO implement
    }
};
}