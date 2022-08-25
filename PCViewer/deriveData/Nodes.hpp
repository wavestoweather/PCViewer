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

// writeable memory view
template<class T>
class memory_view{
    T* _data{};
    size_t _size{};
public:
    memory_view(){};
    memory_view(T& d): _data(&d), _size(1){};
    memory_view(std::vector<T>& v): _data(v.data()), _size(v.size()){};
    template<size_t size>
    memory_view(std::array<T, size>& a): _data(a.data()), _size(size){};
    memory_view(T* data, size_t size): _data(data), _size(size){};
    template<class U>
    memory_view(memory_view<U> m): _data(reinterpret_cast<T*>(m.data())), _size(m.size() * sizeof(U) / sizeof(T)){
        assert(m.size() * sizeof(U) == _size * sizeof(T));   // debug assert to check if the memory views can be converted to each other, e.g. if the element sizes align
    }
    memory_view(const memory_view&) = default;
    memory_view(memory_view&&) = default;
    memory_view& operator=(const memory_view&) = default;
    memory_view& operator=(memory_view&&) = default;

    T* data(){return _data;};
    const T* data() const {return _data;};
    size_t size() const {return _size;};
    bool empty() const {return _size == 0;};
    T& operator[](size_t i){
        assert(i < _size);   // debug assert for in bounds check
        return _data[i];
    }
    const T& operator[](size_t i) const{
        assert(i < _size);
        return _data[i];
    }
    bool operator==(const memory_view& o) const{
        return _data == o._data && _size == o._size;
    }
    operator bool() const {return _data && _size;};

    bool equalData(const memory_view& o) const{
        if(_size != o._size)
            return false;
        for(auto i: irange(_size)){
            if(_data[i] != o._data[i])
                return false;
        }
        return true;
    }

    T* begin() {return _data;};
    T* end() {return _data + _size;};
    const T* begin() const {return _data;};
    const T* end() const {return _data + _size;};
};

template<class T>
struct column_memory_view{ // holds one or more columns (done to also be able to hold vectors)
    memory_view<uint32_t> dimensionSizes{};
    memory_view<uint32_t> columnDimensionIndices{};
    std::vector<memory_view<T>> cols{};

    column_memory_view() = default;
    column_memory_view(memory_view<T> data, memory_view<uint32_t> dimensionSizes = {}, memory_view<uint32_t> columnDimensionIndices = {}):
        dimensionSizes(dimensionSizes),
        columnDimensionIndices(columnDimensionIndices)
        {
            // checking for column or single row data in case of a constant
            if(dimensionSizes.empty()){  // row data
                for(int i: irange(data.size()))
                    cols.push_back(memory_view(data.data() + i, 1));            
            }
            else{
                cols = {data};
            }
        };
    column_memory_view(std::vector<memory_view<T>> dataVec, memory_view<uint32_t> dimensionSizes = {}, memory_view<uint32_t> columnDimensionIndices = {}):
        dimensionSizes(dimensionSizes),
        columnDimensionIndices(columnDimensionIndices),
        cols(dataVec){};

    
    // returns the amount of elements in this column_memory_view
    // Note: diemnsionSizes.empty() indicates a constant in which case the size = 1
    uint64_t size() const{
        uint64_t ret{1};
        for(auto s: irange(dimensionSizes.size())) ret *= dimensionSizes[s];
        return ret;
    }
    // returns if the columns span all dimensions
    bool full() const{
        return size() == cols[0].size();
    }

    bool operator==(const column_memory_view& o) const{
        return dimensionSizes == o.dimensionSizes && columnDimensionIndices == o.columnDimensionIndices && cols == o.cols;
    }
    bool equalData(const column_memory_view& o) const{
        if(cols.size() != o.cols.size())
            return false;
        if(!dimensionSizes.equalData(o.dimensionSizes))
            return false;
        if(!columnDimensionIndices.equalData(o.columnDimensionIndices))
            return false;
        for(auto c: irange(cols)){
            if(!cols[c].equalData(o.cols[c]))
                return false;
        }
        return true;
    }
    // only checks dimensionSizes and columnDimensionIndices for equality
    bool equalDataLayout(const column_memory_view& o) const{
        if(!dimensionSizes.equalData(o.dimensionSizes))
            return false;
        if(!columnDimensionIndices.equalData(o.columnDimensionIndices))
            return false;
        return true;
    }

    T& operator()(uint64_t index, uint32_t column){
        auto cI = columnIndex(index);
        assert(cI < cols[column].size());
        return cols[column][cI];
    }
    const T& operator()(uint64_t index, uint32_t column) const{
        auto cI = columnIndex(index);
        assert(cI < cols[column].size());
        return cols[column][cI];
    }

    operator bool() const{ return cols.size();};
private:
    uint64_t dimensionIndex(const std::vector<uint64_t>& dimensionIndices) const{
        uint32_t columnIndex = 0;
        for(int d = 0; d < columnDimensionIndices.size(); ++d){
            uint32_t factor = 1;
            for(int i = d + 1; i < columnDimensionIndices.size(); ++i){
                factor *= dimensionSizes[columnDimensionIndices[i]];
            }
            columnIndex += factor * dimensionIndices[columnDimensionIndices[d]];
        }
        return columnIndex;
    }
    uint64_t columnIndex(uint64_t index) const{
        std::vector<uint64_t> dimensionIndices(dimensionSizes.size());
        for(int i = dimensionSizes.size() - 1; i >= 0; --i){
            dimensionIndices[i] = index % dimensionSizes[i];
            index /= dimensionSizes[i];
        }
        return dimensionIndex(dimensionIndices);
    }
};

// ------------------------------------------------------------------------------------------
// types
// ------------------------------------------------------------------------------------------
class Type{
public:
    virtual ImVec4 color() const = 0;
    virtual column_memory_view<float> data() = 0;
    ax::Widgets::IconType iconType() const{return ax::Widgets::IconType::Circle;};
};

class FloatType: public Type, public Creatable<FloatType, Type>{
public:
    ImVec4 color() const override{return {1,0,0,1};};
    column_memory_view<float> data() override {return column_memory_view(memory_view(_d));};
private:
    float _d;
};

class ConstantFloatType: public Type, public Creatable<ConstantFloatType, Type>{
    ImVec4 color() const override{return {1, .5, 0, 1};};
    column_memory_view<float> data() override{return {};};    // always returns null pointer as data is not changable
};

class VectorType: public Type,  public Creatable<VectorType, Type>{
public:
    ImVec4 color() const override{return {0,1,0,1};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::vector<float> _d;
};

class Vec2Type: public Type,  public Creatable<Vec2Type, Type>{
public:
    ImVec4 color() const override{return {.5, .5, .5, 1};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::array<float, 2> _d;
};

class Vec3Type: public Type,  public Creatable<Vec3Type, Type>{
public:
    ImVec4 color() const override{return {1,1,1,1};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::array<float, 3> _d;
};

class Vec4Type: public Type,  public Creatable<Vec4Type, Type>{
public:
    ImVec4 color() const override{return {.1, .1, .1, 1};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::array<float, 4> _d;
};

// ------------------------------------------------------------------------------------------
// nodes
// ------------------------------------------------------------------------------------------
using float_column_views = std::vector<column_memory_view<float>>;
class Node{
public:
    std::vector<std::unique_ptr<Type>> inputTypes;
    std::vector<std::string> inputNames;
    std::vector<std::unique_ptr<Type>> outputTypes;
    std::vector<std::string> outputNames;

    std::string name;
    std::string middleText;

    Node(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {}):
        inputTypes(std::move(inputTypes)),
        inputNames(inputNames),
        outputTypes(std::move(outputTypes)),
        outputNames(outputNames),
        name(header),
        middleText(mt){}

    virtual int outputChannels() const { uint32_t count{}; for(const auto& t: outputTypes) count += t->data().cols.size();return count;};
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const = 0;
};

struct NodesRegistry{
    struct Entry{
        std::unique_ptr<Node> prototype;
        std::function<std::unique_ptr<Node>()> create;
    };
    static std::map<std::string, Entry> nodes;
    NodesRegistry(std::string name, std::function<std::unique_ptr<Node>()> createFunction) {if(nodes.count(name) == 0) nodes[name] = {createFunction(), createFunction};};
};

// registers the nodes with a standard constructor
#define REGISTER_NODE(class) static NodesRegistry classReg_##class(#class , class::create<>);

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
        if(input[0].size() < 100){
            for(int i: irange(input[0].size())){
                if(input[0].cols.size() > 1)
                    std::cout << "(";
                for(int j: irange(input[0].cols))
                    std::cout << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    std::cout << "), ";
                if(i % 20 == 0)
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
                if(i % 20 == 0)
                    std::cout << std::endl;
            }
            std::cout << std::endl << "  ...  " << std::endl;
            for(int i: irange(input[0].size() - 50, input[0].size())){
                if(input[0].cols.size() > 1)
                    std::cout << "(";
                for(int j: irange(input[0].cols))
                    std::cout << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    std::cout << "), ";
                if(i % 20 == 0)
                    std::cout << std::endl;
            }
        }
        std::cout << "]" << std::endl;
    };
};

class VariableInput{  // simple class to indicate variable input lengths
public:
    int minNodes;
    int maxNodes;
    
    VariableInput(int minInputs = 0, int maxInputs = std::numeric_limits<int>::max()): minNodes(minInputs), maxNodes(maxInputs){}
};

class DatasetOutputNode: public OutputNode, VariableInput, public Creatable<DatasetOutputNode>{
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