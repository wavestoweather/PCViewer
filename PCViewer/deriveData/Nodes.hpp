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
#include <set>
#include "../util/ranges.hpp"
#include "../tsne/tsne.h"
#include <Eigen/Dense>
#include <robin_hood.h>
#include <radix.hpp>
#include <sstream>
#include <kd_tree.hpp>
#include <flat_set.hpp>
#include "k_means.hpp"
#include "db_scan.hpp"
#include "gpu_instructions.hpp"

namespace deriveData{
namespace Nodes{
// convenience function
inline void applyNonaryFunction(const float_column_views& input, float_column_views& output, uint32_t col, std::function<float()> f){
    assert(input[0].dimensionSizes.empty());            // check for single value constant in input
    assert(input[0].cols.size() && output[0].cols.size()); // check for columns
    assert(output[0].size() == input[0].cols[col][0]);    // enough memory has to be allocated before this call is made..
    for(size_t i: util::size_range(output[0]))
        output[0].cols[0][i] = f();
};

inline void applyUnaryFunction(const float_column_views& input, float_column_views& output, uint32_t col, std::function<float(float)> f){
    assert(input[0].equalDataLayout(output[0]));
    for(size_t i: util::size_range(input[0].cols[0]))
        output[0].cols[col][i] = f(input[0].cols[col][i]);
};

inline void applyMultiDimUnaryFunction(const float_column_views& input, float_column_views& output, const std::vector<uint32_t>& dims, std::function<float(const std::vector<float>&)> f){
    assert(input[0].equalDataLayout(output[0]));
    std::vector<float> in(dims.size());
    for(size_t i: util::size_range(output[0].cols[0])){
        for(int j: irange(dims))
            in[j] = input[0].cols[dims[j]][i];
        output[0].cols[0][i] = f(in);
    }
};

inline void applyNAryFunction(const float_column_views& input, float_column_views& output, std::function<float(const std::vector<float>&)> f){
    assert(input[0].equalDataLayout(output[0]));
    std::vector<float> in(input.size());
    for(size_t i: util::size_range(output[0].cols[0])){
        for(size_t j: util::size_range(input))
            in[j] = input[j].cols[0][i];
        output[0].cols[0][i] = f(in);
    }
}

inline float unaryReductionFunction(const float_column_views& input, uint32_t col, float initVal, std::function<float(float, float)> f){
    for(size_t i: util::size_range(input[0].cols[col]))
        initVal = f(initVal, input[0].cols[i]);
    return initVal;
};

// expects indices at input[0]
inline void applyUnaryReductionFunction(const float_column_views& input, float init_value, column_memory_view<float>& result, std::function<float(float, float)> combine_f, std::function<float(float, size_t)> finish_f = [](float a, size_t){return a;}){
    // init values
    struct per_group_data{float combine_val; size_t count;};
    //std::map<size_t, per_group_data> reduction_data;
    robin_hood::unordered_map<size_t, per_group_data> reduction_data;
    if(input[1].equalDataLayout(input[0])){
        for(size_t i: util::size_range(input[1].cols[0])){
            size_t group_index = size_t(input[0].cols[0][i]);
            bool contained = reduction_data.count(group_index);
            auto& e = reduction_data[group_index];
            if(!contained)
                e = {init_value, 0};
            e.combine_val = combine_f(e.combine_val, input[1].cols[0][i]);
            ++e.count;
        }
    }
    else{
        for(size_t i: util::size_range(input[1])){
            size_t group_index = static_cast<size_t>(input[0](i, 0));
            bool contained = reduction_data.count(group_index);
            auto& e = reduction_data[group_index];
            if(!contained)
                e = {init_value, 0};
            e.combine_val = combine_f(e.combine_val, input[1](i, 0));
            ++e.count;
        }
    }
    assert(input[0].equalDataLayout(result));  // copying the reduced values to the result view. Automatically takes multi dimensional indices into account
    for(size_t i: util::size_range(input[0].cols[0])){
        auto& d = reduction_data[size_t(input[0].cols[0][i])];
        result.cols[0][i] = finish_f(d.combine_val, d.count);
    }
}

inline void applyBinaryFunction(const float_column_views& input, float_column_views& output, uint32_t col, std::function<float(float, float)> f){
    assert(input[0].size() == output[0].size() || input[1].size() == output[0].size()); // only one needs to have the same size
    if(input[0].equalDataLayout(input[1]) && input[0].equalDataLayout(output[0])){      // fast operation possible
        for(size_t i: util::size_range(output[0].cols[col])){
            output[0].cols[col][i] = f(input[0].cols[col][i], input[1].cols[col][i]);
        }
    }
    else{                                                                               // less efficient but more general
        for(size_t i: util::size_range(output[0])){
            output[0](i, col) = f(input[0](i, col), input[1](i, col));
        }
    }
};

inline std::tuple<float, float> binaryReductionFunction(const float_column_views& input, uint32_t col, std::tuple<float, float> initVal, std::function<std::tuple<float, float>(std::tuple<float, float>, float)> f){
    for(size_t i: util::size_range(input[0].cols[col]))
        initVal = f(initVal, input[0].cols[col][i]);
    return initVal;
};

inline void tryAlignInputOutput(const float_column_views& input, float_column_views& output, uint32_t viewIndex = 0){
    assert(input.size() == output.size());
    assert(input[viewIndex].equalDataLayout(output[viewIndex]));
    assert(input[viewIndex].cols.size() == output[viewIndex].cols.size());
    if(input[viewIndex].cols.size() == 1)
        return;
    std::vector<int> preferredPlace(output[viewIndex].cols.size(), -1);
    // getting the preferred place
    for(size_t i: util::size_range(output[viewIndex].cols)){
        for(size_t j: util::size_range(input[viewIndex].cols)){
            if(input[viewIndex].cols[j] == output[viewIndex].cols[i]){
                preferredPlace[i] = static_cast<int>(j);
                break;
            }
        }
    }
    // eliminate injective places (multiple outputs want to be matched to the same input)
    std::set<int> takenSpaces;
    for(int& i: preferredPlace){
        if(i != -1){
            if(takenSpaces.count(i) > 0)
                i = -1;
            if(i != -1)
                takenSpaces.insert(i);
        }
    }
    // switching to the preferred place
    float_column_views newOutput(1);
    newOutput[0].dimensionSizes = output[viewIndex].dimensionSizes;
    newOutput[0].columnDimensionIndices = output[viewIndex].columnDimensionIndices;
    newOutput[0].cols.resize(output[viewIndex].cols.size());
    for(size_t i: util::size_range(preferredPlace)){
        if(preferredPlace[i] == -1) // no preferred place, skip
            continue;
        newOutput[0].cols[preferredPlace[i]] = output[viewIndex].cols[i];
    }
    int currentIndex = 0;
    for(size_t i: util::size_range(preferredPlace)){
        if(preferredPlace[i] != -1) // was already added, skip
            continue;
        while(newOutput[0].cols[currentIndex]) ++currentIndex;  // get next free spot
        newOutput[0].cols[currentIndex++] = output[viewIndex].cols[i];
    }
    output[viewIndex] = newOutput[0];
};

// ------------------------------------------------------------------------------------------
// nodes
// ------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------
// special nodes
// ------------------------------------------------------------------------------------------

class Input{
};

class Output{
};

class DatasetInput: public Node, public Input, public Creatable<DatasetInput>{
public:
    std::string_view datasetId;

    DatasetInput(std::string_view datasetID = {""}):
        Node(createFilledVec<FloatType, Type>(0), {}, createFilledVec<FloatType, Type>(0),{}, "", ""), datasetId(datasetID)
    {
        name = "Dataset Input";
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // Nothing to do. The data is simply forwarded via a view
    };
};

class DataCreation: public Node, public Input{
public:
    DataCreation(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {}): Node(std::move(inputTypes), std::move(inputNames), std::move(outputTypes), std::move(outputNames), header, mt){};
};

class Vector_Zero: public DataCreation, public Creatable<Vector_Zero>{
public:
    Vector_Zero(): DataCreation(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "Zero Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyNonaryFunction(input, output, 0, [](){return 0;});
    };

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::zero_vec, input, output);
        output[0].cols_min_max = {min_max_t{0.f,0.f}};
    }
};

class Vector_One: public DataCreation, public Creatable<Vector_One>{
public:
    Vector_One(): DataCreation(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "One Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyNonaryFunction(input, output, 0, [](){return 1;});
        output[0].cols_min_max = {min_max_t{1.f,1.f}};
    };

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::one_vec, input, output);
        output[0].cols_min_max = {min_max_t{1.f,1.f}};
    }
};

class Vector_Random: public DataCreation, public Creatable<Vector_Random>{
public:
    Vector_Random(): DataCreation(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "Random Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyNonaryFunction(input, output, 0, [](){return double(rand()) / RAND_MAX;});
        output[0].cols_min_max = {min_max_t{0.f,1.f}};
    };

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::rand_vec, input, output);
        output[0].cols_min_max = {min_max_t{0.f,1.f}};
    }
};

class Vector_Iota: public DataCreation, public Creatable<Vector_Iota>{
public:
    Vector_Iota(): DataCreation(createFilledVec<FloatType, Type>(1), {"Size"}, createFilledVec<FloatType, Type>(1), {""}, "Iota Vector", ""){};

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        size_t cur_index{};
        applyNonaryFunction(input, output, 0, [&cur_index](){return cur_index++;});
        output[0].cols_min_max = {min_max_t{0.f, input[0].cols[0][0] - 1.f}};
    };

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::iota_vec, input, output);
        output[0].cols_min_max = {min_max_t{0.f, input[0].cols[0][0] - 1.f}};
    }
};

class Active_Indices: public DataCreation, public Creatable<Active_Indices>{
public:
    Active_Indices(): DataCreation({}, {}, createFilledVec<IndexType, Type>(1), {""}, "Active Indices") 
    {
        input_elements[middle_input_id]["Drawlist/Templatelist"] = drawlist_templatelist_selector;
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        switch(input[0].cols.size()){
        case 1:{
            if(input[0].cols[0].empty())
                std::fill(output[0].cols[0].begin(), output[0].cols[0].end(), 1.f);
            else{
                memory_view<const uint32_t> indices{reinterpret_cast<const uint32_t*>(input[0].cols[0].data()), input[0].cols[0].size()};
                std::fill(output[0].cols[0].begin(), output[0].cols[0].end(), .0f);
                for(uint32_t i: indices)
                    output[0].cols[0][i] = 1.f;
            }
            break;}
        case 2:{
            memory_view<const uint32_t> activations{reinterpret_cast<const uint32_t*>(input[0].cols[0].data()), input[0].cols[0].size()};
            memory_view<const uint32_t> indices{reinterpret_cast<const uint32_t*>(input[0].cols[1].data()), input[0].cols[1].size()};
            if(indices.empty()){
                for(size_t i: util::i_range(input[0].size())){
                    if((activations[i / 32] >> (i % 32)) & 1) 
                        output[0].cols[0][i] = 1.f;
                    else
                        output[0].cols[0][i] = 0.f;
                }
            }
            else{
                std::fill(output[0].cols[0].begin(), output[0].cols[0].end(), .0f);
                for(size_t i: util::size_range(input[0].cols[1])){
                    if((activations[i / 32] >> (i % 32)) & 1)
                        output[0].cols[0][indices[i]] = 1.f;
                }
            }
            break;}
        default:
            throw std::runtime_error{"deriveData::Nodes::Active_Indices() Unsupported amount of input elements"};
        }
        output[0].cols_min_max = {min_max_t{0.f,1.f}};
    };
};

class Serialization: public Node{
public:
    Serialization(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {}): Node(std::move(inputTypes), std::move(inputNames), std::move(outputTypes), std::move(outputNames), header, mt) {}

    std::string serialize(const float_column_views& input) const{
        // prints at max the first 50 and last 50 items of a vector
        std::stringstream out;
        std::string out_name = input_elements[middle_input_id]["Output Name"].get<std::string>();
        if(out_name.size())
            out << out_name << ": ";
        out << "[ ";
        if(input[0].columnSize() < 100){
            for(size_t i: util::i_range(input[0].columnSize())){
                if(input[0].cols.size() > 1)
                    out << "(";
                for(size_t j: util::size_range(input[0].cols))
                    out << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    out << "), ";
                if((i + 1) % 20 == 0)
                    out << "\n";
            }
        }
        else{
            for(int i: util::i_range(50)){
                 if(input[0].cols.size() > 1)
                    out << "(";
                for(size_t j: util::size_range(input[0].cols))
                    out << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    out << "), ";
                if((i + 1) % 20 == 0)
                    out << "\n";
            }
            out << "\n" << "  ...  " << "\n";
            for(size_t i: util::i_range(input[0].columnSize() - 50, input[0].columnSize())){
                if(input[0].cols.size() > 1)
                    out << "(";
                for(size_t j: util::size_range(input[0].cols))
                    out << input[0].cols[j][i] << ", ";
                if(input[0].cols.size() > 1)
                    out << "), ";
                if((i + 1) % 20 == 0)
                    out << "\n";
            }
        }
        out << "] (size: " << input[0].size() << ", column size: " << input[0].columnSize() << ", data range:[" << input[0].cols_min_max[0].min << "," << input[0].cols_min_max[0].max << "])";
        return out.str();
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::copy, input, output);
    }
};

class Print_Vector: public Output, public Serialization, public Creatable<Print_Vector>{
public:
    Print_Vector(): Serialization(createFilledVec<FloatType, Type>(1), {""}, createFilledVec<FloatType, Type>(0), {}, "Print Vector"){
        input_elements[middle_input_id]["Output Name"] = "";
    }

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // nothing to do
    }
};

class Print_Indices: public Output, public Serialization, public Creatable<Print_Indices>{
public:
    Print_Indices(): Serialization(createFilledVec<IndexType, Type>(1), {""}, createFilledVec<FloatType, Type>(0), {}, "Print Indices"){
        input_elements[middle_input_id]["Output Name"] = "";
    }

    void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // nothing to do
    }
};

class VariableInput{  // simple class to indicate variable input lengths
public:
    int minNodes;
    int maxNodes;
    bool namedInput{true};
    
    VariableInput(bool namedInput = true, int minInputs = 0, int maxInputs = std::numeric_limits<int>::max()):namedInput(namedInput), minNodes(minInputs), maxNodes(maxInputs){}

    virtual void pinAddAction(){}
    virtual void pinRemoveAction(int i){}
};

class VariableOutput{
public:
    int minOutputs;
    int maxOutputs;

    VariableOutput(int minOutputs = 0, int maxOutputs = std::numeric_limits<int>::max()): minOutputs(minOutputs), maxOutputs(maxOutputs) {}
};

class DatasetOutput: public Node, public Output, public VariableInput, public Creatable<DatasetOutput>{
public:
    std::string_view datasetId;

    DatasetOutput(std::string_view datasetID = {""}):
        Node(createFilledVec<FloatType, Type>(0), {}, createFilledVec<FloatType, Type>(0),{}, "", ""), datasetId(datasetID)
    {
        name = "Dataset Output";
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // Nothing to do. The data has to be moved outside of the node as we need teh vector storing the data to move the data into the dataset
    };
};

class Sum: public Node, public VariableInput, public Creatable<Sum>{
public:
    Sum(): 
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<FloatType, Type>(1), {""}, "Sum"),
        VariableInput(false, 1)
    {
        input_elements[input_input_id] = crude_json::array{1., 1.};
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        auto& prefactors = input_elements[input_input_id].get<crude_json::array>();
        assert(prefactors.size() == input.size());
        applyNAryFunction(input, output, 
            [&prefactors](const std::vector<float>& v) {
                float res{};
                for(size_t i: util::size_range(v))
                    res += static_cast<float>(prefactors[i].get<double>() * v[i]);
                return res;
            }
        );
        output[0].cols_min_max = {min_max_t{0.f, 0.f}};
        for(const auto& in: input){
            output[0].cols_min_max[0].min += in.cols_min_max[0].min;
            output[0].cols_min_max[0].max += in.cols_min_max[0].max;
        }
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::sum, input, output, input_elements[input_input_id]);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min + input[1].cols_min_max[0].min, input[0].cols_min_max[0].min + input[1].cols_min_max[0].min}};
        output[0].cols_min_max = {min_max_t{0.f, 0.f}};
        for(const auto& in: input){
            output[0].cols_min_max[0].min += in.cols_min_max[0].min;
            output[0].cols_min_max[0].max += in.cols_min_max[0].max;
        }
    }

    void pinAddAction() override {input_elements[input_input_id].get<crude_json::array>().push_back(1.);}
    void pinRemoveAction(int i) override {input_elements[input_input_id].get<crude_json::array>().erase(input_elements[input_input_id].get<crude_json::array>().begin() + i);}
};

class Product: public Node, public VariableInput, public Creatable<Product>{
public:
    Product(): 
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<FloatType, Type>(1), {""}, "Product"),
        VariableInput(false, 1)
    {
        input_elements[input_input_id] = crude_json::array{1., 1.};
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        auto& exponents = input_elements[input_input_id].get<crude_json::array>();
        assert(exponents.size() == input.size());
        applyNAryFunction(input, output, 
            [&exponents](const std::vector<float>& v) {
                double res{1.};
                for(size_t i: util::size_range(v))
                    res *= std::pow(v[i], exponents[i].get<double>());
                return float(res);
            }
        );
        output[0].cols_min_max = {min_max_t{1.f, 1.f}};
        for(const auto& in: input){
            output[0].cols_min_max[0].min *= in.cols_min_max[0].min;
            output[0].cols_min_max[0].max *= in.cols_min_max[0].max;
        }
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::product, input, output, input_elements[input_input_id]);
        output[0].cols_min_max = {min_max_t{1.f, 1.f}};
        for(const auto& in: input){
            output[0].cols_min_max[0].min *= in.cols_min_max[0].min;
            output[0].cols_min_max[0].max *= in.cols_min_max[0].max;
        }
    }

    void pinAddAction() override {input_elements[input_input_id].get<crude_json::array>().push_back(1.);}
    void pinRemoveAction(int i) override {input_elements[input_input_id].get<crude_json::array>().erase(input_elements[input_input_id].get<crude_json::array>().begin() + i);}
};

class Lp_Norm: public Node, public VariableInput, public Creatable<Lp_Norm>{
public:
    Lp_Norm():
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<FloatType, Type>(1), {""}, "Lp-Norm"),
        VariableInput(false, 1)
    {
        input_elements[middle_input_id]["Lp-Norm"] = 2.;
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        float exp = static_cast<float>(input_elements[middle_input_id]["Lp-Norm"].get<double>());
        applyNAryFunction(input, output, 
            [&exp](const std::vector<float>& v){
                float res{};
                for(float i: v)
                    res += std::pow(std::abs(i), exp);
                return std::pow(res, 1./exp);
            }
        );
        float res{};
        for(const auto& in: input) res += std::pow(std::abs(in.cols_min_max[0].min), exp);
        output[0].cols_min_max[0].min = std::pow(res, 1./exp);
        res = {};
        for(const auto& in: input) res += std::pow(std::abs(in.cols_min_max[0].max), exp);
        output[0].cols_min_max[0].min = std::pow(res, 1./exp);
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::lp_norm, input, output, input_elements[middle_input_id]);
        float res{};
        for(const auto& in: input) res += std::pow(std::abs(in.cols_min_max[0].min), exp);
        output[0].cols_min_max[0].min = std::pow(res, 1./exp);
        res = {};
        for(const auto& in: input) res += std::pow(std::abs(in.cols_min_max[0].max), exp);
        output[0].cols_min_max[0].min = std::pow(res, 1./exp);
    }
};

class PCA_Projection: public Node, public VariableInput, public VariableOutput, public Creatable<PCA_Projection>{
public:
    PCA_Projection():
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<FloatType, Type>(2), {"", ""}, "PCA-Projection"),
        VariableInput(false, 1),
        VariableOutput(1)
    {}

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        // convert data to 
        Eigen::MatrixXf m(input[0].size(), input.size());
        if(equalDataLayouts<float>(input)){
            for(size_t i: util::size_range(input)){
                for(size_t j: util::size_range(input[i].cols[0]))
                    m(j, i) = input[i].cols[0][j];
            }
        }
        else{
            for(size_t i: util::size_range(input)){
                for(size_t j: util::i_range(input[i].size()))
                    m(j, i) = input[i](j, 0);
            }
        }
        
        auto mean_cols = m.colwise().mean();
        m = m.rowwise() - mean_cols;
        float min = m.minCoeff();
        float max = m.maxCoeff();
        float diff = std::max(max, std::abs(min));
        m /= diff;
        
        // caclulating the PCA
        Eigen::BDCSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU);
        m = svd.matrixU().real() * svd.singularValues().real().asDiagonal();
        
        // transferring to teh output
        for(size_t i: util::size_range(output)){
            for(size_t j: util::size_range(output[i].cols[0]))
                output[i].cols[0][j] = m(j, i);
        }

        // TODO proper output min max
        for(auto& out: output)
            out.cols_min_max[0] = input[0].cols_min_max[0];
    }
};

class TSNE_Projection: public Node, public VariableInput, public VariableOutput, public Creatable<TSNE_Projection>{
public:
    TSNE_Projection():
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<FloatType, Type>(2), {"", ""}, "TSNE-Projection"),
        VariableInput(false, 1),
        VariableOutput(1)
    {
        input_elements[middle_input_id]["perplexity"] = 30.;
        input_elements[middle_input_id]["theta"] = .5;
        input_elements[middle_input_id]["random seed (negative for none)"] = -1.;
        input_elements[middle_input_id]["skip random init"] = false;
        input_elements[middle_input_id]["iterations"] = 500.;
        input_elements[middle_input_id]["stop lying iteration"] = 700.;
        input_elements[middle_input_id]["momentum switch iteration"] = 700.;
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        std::vector<util::memory_view<const float>> in(input.size());
        std::vector<util::memory_view<float>> out(output.size());
        std::vector<std::vector<float>> inflated_in;
        std::atomic<float> progress{};

        if(equalDataLayouts<float>(input)){
            for(size_t i: util::size_range(input))
                in[i] = util::memory_view<const float>{input[i].cols[0].data(), input[i].cols[0].size()};
        }
        else{
            for(size_t i: util::size_range(input)){
                inflated_in.emplace_back(output[0].size());
                for(size_t j: util::size_range(output[0].cols[0])){
                    const auto indices = output[0].columnIndexToDimensionIndices(j);
                    inflated_in.back()[j] = input[i].cols[0][input[i].dimensionIndicesToColumnIndex(indices)];
                }
                in[i] = util::memory_view<const float>{inflated_in.back().data(), inflated_in.back().size()};
            }
        }
        for(size_t i: util::size_range(output))
            out[i] = util::memory_view<float>{output[i].cols[0].data(), output[i].cols[0].size()};
        TSNE::run_cols(in, out, input_elements[middle_input_id]["perplexity"].get<double>(), 
                                input_elements[middle_input_id]["theta"].get<double>(),
                                int(input_elements[middle_input_id]["random seed (negative for none)"].get<double>()),
                                input_elements[middle_input_id]["skip random init"].get<bool>(),
                                int(input_elements[middle_input_id]["iterations"].get<double>()),
                                int(input_elements[middle_input_id]["stop lying iteration"].get<double>()),
                                int(input_elements[middle_input_id]["momentum switch iteration"].get<double>()),
                                progress);

        // TODO proper min max values
        for(auto& out: output)
            out.cols_min_max[0] = input[0].cols_min_max[0];
    }
};

class K_Means: public Node, public VariableInput, public Creatable<K_Means>{
public:
    K_Means():
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<IndexType, Type>(1), {""}, "K Means Clustering", {}, false),
        VariableInput(false, 1)
    {
        util::json::add_enum(input_elements[middle_input_id], "Distance Method", k_means::distance_method_names, k_means::distance_method_t::norm);

        input_elements[middle_input_id]["Amount of cluster"] = 5.;

        util::json::add_enum(input_elements[middle_input_id], "Init method", k_means::init_method_names, k_means::init_method_t::plus_plus);
        util::json::add_enum(input_elements[middle_input_id], "K Means Method", k_means::mean_method_names, k_means::mean_method_t::mean);

        input_elements[middle_input_id]["Max Iterations"] = 20.;
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        k_means::k_means_settings_t settings;
        settings.distance_method = util::json::get_enum_val<k_means::distance_method_t>(input_elements[middle_input_id]["Distance Method"]);
        settings.cluster_count = static_cast<int>(input_elements[middle_input_id]["Amount of cluster"].get<double>());
        settings.init_method = util::json::get_enum_val<k_means::init_method_t>(input_elements[middle_input_id]["Init method"]);
        settings.mean_method = util::json::get_enum_val<k_means::mean_method_t>(input_elements[middle_input_id]["K Means Method"]);
        settings.max_iteration = static_cast<int>(input_elements[middle_input_id]["Max Iterations"].get<double>());
        k_means::run(input, output, settings);

        output[0].cols_min_max = {min_max_t{0, float(settings.cluster_count - 1)}};
    }
};

class DB_Scan: public Node, public VariableInput, public Creatable<DB_Scan>{
public:
    DB_Scan():
        Node(createFilledVec<FloatType, Type>(2), {"", ""}, createFilledVec<IndexType, Type>(1), {""}, "DB-Scan Clustering", {}, false),
        VariableInput(false, 1)
    {
        input_elements[middle_input_id]["Epsilon"] = .1;
        input_elements[middle_input_id]["Min Points"] = 5.;
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        db_scan::db_scans_settings_t settings{};
        settings.epsilon = static_cast<float>(input_elements[middle_input_id]["Epsilon"].get<double>());
        settings.min_points = static_cast<int>(input_elements[middle_input_id]["Min Points"].get<double>());

        int max_cluster = db_scan::run(input, output, settings);

        output[0].cols_min_max = {min_max_t{0, float(max_cluster - 1)}};
    }
};

class Group_Distance: public Node, public VariableInput, public Creatable<Group_Distance>{
public:
    Group_Distance():
        Node(createFilledVec<IndexType, Type>(2), {"", ""}, createFilledVec<FloatType, Type>(1), {""}, "Group Distance"),
        VariableInput(false, 1)
    {
        inputTypes[1] = std::make_unique<FloatType>();
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        // creating the KDtree for fast data lookup
        std::vector<uint32_t> indices;
        for(size_t i: util::size_range(input[0].cols[0])) if(input[0].cols[0][i] > 0) indices.emplace_back(static_cast<uint32_t>(i));
        const auto data_input = std::vector<column_memory_view<float>>(input.begin() + 1, input.end());
        const structures::kd_tree tree(data_input, indices, input[0].equalDataLayout(input[1]));
        for(size_t i: util::size_range(input[0].cols[0])){
            //size_t col_ind = 
            auto [n, dist] = tree.nearest_neighbour(i);
            output[0].cols[0][i] = std::sqrt(dist);
        }

        // TODO proper value
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0]}};
    }
};

// ------------------------------------------------------------------------------------------
// unary nodes
// ------------------------------------------------------------------------------------------

template<class T, class T_out = T>
class Unary: public Node{
public:
    Unary(std::string_view header, std::string_view middle):
        Node(createFilledVec<T, Type>(1), {std::string()}, createFilledVec<T_out, Type>(1),{std::string()}, header, middle){};
};

class Cast_to_Float: public Unary<IndexType, FloatType>, public Creatable<Cast_to_Float>{
public:
    Cast_to_Float(): Unary("", "cast_to<float[]>") {}

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{     
        if(input[0].cols[0] == output[0].cols[0])
            return;
        for(size_t i: util::size_range(input[0].cols[0]))
            output[0].cols[0][i] = input[0].cols[0][i];
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0]}};
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        if(input[0].cols[0] != output[0].cols[0])
            add_operation(operations, op_codes::copy, input, output);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0]}};
    }
};

class Cast_to_Index: public Unary<FloatType, IndexType>, public Creatable<Cast_to_Index>{
public:
    Cast_to_Index(): Unary("", "cast_to<index[]>") {}

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{     
        for(size_t i: util::size_range(input[0].cols[0]))
            output[0].cols[0][i] = std::floor(input[0].cols[0][i]);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0]}};
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        if(input[0].cols[0] != output[0].cols[0])
            add_operation(operations, op_codes::copy, input, output);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0]}};
    }
};

class Inverse: public Unary<FloatType>, public Creatable<Inverse>{
public:
    Inverse(): Unary("", "1/"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{     
        applyUnaryFunction(input, output, 0, [](float in){return 1. / in;});
        output[0].cols_min_max = {min_max_t{1.f / input[0].cols_min_max[0].max, 1.f / input[0].cols_min_max[0].min}};
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::inverse, input, output);
        output[0].cols_min_max = {min_max_t{1.f / input[0].cols_min_max[0].max, 1.f / input[0].cols_min_max[0].min}};
    }
};

class Negate: public Unary<FloatType>, public Creatable<Negate>{
public:
    Negate(): Unary("", "*-1"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{        
        applyUnaryFunction(input, output, 0, [](float in){return -in;});
        output[0].cols_min_max = {min_max_t{-input[0].cols_min_max[0].max, -input[0].cols_min_max[0].min}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::negate, input, output);
        output[0].cols_min_max = {min_max_t{-input[0].cols_min_max[0].max, -input[0].cols_min_max[0].min}};
    }
};

class Normalization: public Unary<FloatType>, public Creatable<Normalization>{
public:
    enum class NormalizationType{
        ZeroOne,
        MinusOneOne
    };
    NormalizationType normalizationType{NormalizationType::ZeroOne};

    Normalization(): Unary("", "norm"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        auto [min, max] = binaryReductionFunction(input, 0, {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}, [](std::tuple<float, float> prev, float cur){return std::tuple<float, float>{std::min(std::get<0>(prev), cur), std::max(std::get<1>(prev), cur)};});
        if(min == max){
            max += 1.f;
            max *= 1.1f;
        }
        float mi = min, ma = max;
        applyUnaryFunction(input, output, 0, [mi, ma](float in){return (in - mi) / (ma - mi);});

        output[0].cols_min_max = {min_max_t{normalizationType == NormalizationType::ZeroOne ? 0.f: -1.f, 1.f}};
    }
};

class AbsoluteValue: public Unary<FloatType>, public Creatable<AbsoluteValue>{
public:
    AbsoluteValue(): Unary("", "abs"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return std::abs(in);});
        
        float min_x_max = input[0].cols_min_max[0].min * input[0].cols_min_max[0].max;
        output[0].cols_min_max = {min_max_t{std::abs(input[0].cols_min_max[0].min), std::abs(input[0].cols_min_max[0].max)}};
        if(output[0].cols_min_max[0].min > output[0].cols_min_max[0].max)
            std::swap(output[0].cols_min_max[0].min, output[0].cols_min_max[0].max);
        if(min_x_max < 0)
            output[0].cols_min_max[0].min = 0;
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::abs, input, output);

        float min_x_max = input[0].cols_min_max[0].min * input[0].cols_min_max[0].max;
        output[0].cols_min_max = {min_max_t{std::abs(input[0].cols_min_max[0].min), std::abs(input[0].cols_min_max[0].max)}};
        if(output[0].cols_min_max[0].min > output[0].cols_min_max[0].max)
            std::swap(output[0].cols_min_max[0].min, output[0].cols_min_max[0].max);
        if(min_x_max < 0)
            output[0].cols_min_max[0].min = 0;
    }
};

class Square: public Unary<FloatType>, public Creatable<Square>{
public:
    Square(): Unary("", "square"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return in * in;});
        float min_x_max = input[0].cols_min_max[0].min * input[0].cols_min_max[0].max;
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min * input[0].cols_min_max[0].min, input[0].cols_min_max[0].max * input[0].cols_min_max[0].max}};
        if(output[0].cols_min_max[0].min > output[0].cols_min_max[0].max)
            std::swap(output[0].cols_min_max[0].min, output[0].cols_min_max[0].max);
        if(min_x_max < 0)
            output[0].cols_min_max[0].min = 0;
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::square, input, output);
        float min_x_max = input[0].cols_min_max[0].min * input[0].cols_min_max[0].max;
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min * input[0].cols_min_max[0].min, input[0].cols_min_max[0].max * input[0].cols_min_max[0].max}};
        if(output[0].cols_min_max[0].min > output[0].cols_min_max[0].max)
            std::swap(output[0].cols_min_max[0].min, output[0].cols_min_max[0].max);
        if(min_x_max < 0)
            output[0].cols_min_max[0].min = 0;
    }
};

class Sqrt: public Unary<FloatType>, public Creatable<Sqrt>{
public:
    Sqrt(): Unary("", "sqrt"){};

    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return std::sqrt(in);});
        output[0].cols_min_max = {min_max_t{std::sqrt(input[0].cols_min_max[0].min), std::sqrt(input[0].cols_min_max[0].max)}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::sqrt, input, output);
        output[0].cols_min_max = {min_max_t{std::sqrt(input[0].cols_min_max[0].min), std::sqrt(input[0].cols_min_max[0].max)}};
    }
};

class Exponential: public Unary<FloatType>, public Creatable<Exponential>{
public:
    Exponential(): Unary("", "exp"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return std::exp(in);});
        output[0].cols_min_max = {min_max_t{std::exp(input[0].cols_min_max[0].min), std::exp(input[0].cols_min_max[0].max)}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::exp, input, output);
        output[0].cols_min_max = {min_max_t{std::exp(input[0].cols_min_max[0].min), std::exp(input[0].cols_min_max[0].max)}};
    }
};

class Logarithm: public Unary<FloatType>, public Creatable<Logarithm>{
public:
    Logarithm(): Unary("", "log"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return std::log(in);});
        output[0].cols_min_max = {min_max_t{std::log(input[0].cols_min_max[0].min), std::log(input[0].cols_min_max[0].max)}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::log, input, output);
        output[0].cols_min_max = {min_max_t{std::log(input[0].cols_min_max[0].min), std::log(input[0].cols_min_max[0].max)}};
    }
};

class Value_to_Index: public Unary<FloatType, IndexType>, public Creatable<Value_to_Index>{
public:
    Value_to_Index(): Unary("", "Value to Index") {}

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // sorting the input array
        std::vector<float> cpy(input[0].cols[0].begin(), input[0].cols[0].end());
        auto [b, e] = radix::sort(cpy.data(), cpy.data() + cpy.size(), output[0].cols[0].begin());
        robin_hood::unordered_map<float, uint32_t> val_to_index;
        uint32_t cur_index{};
        for(auto i = b; i != e; ++i){
            if(!val_to_index.contains(*i))
                val_to_index[*i] = cur_index++;
        }
        for(size_t i: util::size_range(input[0].cols[0]))
            output[0].cols[0][i] = static_cast<float>(val_to_index[input[0].cols[0][i]]);
        //TODO proper min max
        output[0].cols_min_max = {min_max_t{0, float(input[0].cols[0].size())}};
    }
};

//class ViewTransformNode{};  // simple type to signal that the deriving class does a view transformation to signal that for dataset input nodes an out pin with outputCounts == 1 can be forwared // TODO complex interactions are at work, think about

// currently unused vector types. Easier to make all the more multidim things with variable einputs
class UnaryVec2: public Unary<Vec2Type>{
public:
    UnaryVec2(std::string_view header = "", std::string_view body = ""): Unary(header, body){};
};

class CreateVec2: public UnaryVec2, public Creatable<CreateVec2>{
public:
    CreateVec2(): UnaryVec2(){
        inputTypes = createFilledVec<FloatType, Type>(2);
        inputNames = {"x", "y"};
        outputNames = {"vec2"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // if different memory addresses, copy
        assert(input.size() == 2 && input[0].cols.size() == 1 && input[1].cols.size() == 1 && output.size() == 1 && output[0].cols.size() == 2);
        const float_column_views inputPaired = {column_memory_view<float>{input[0].dimensionSizes, input[0].columnDimensionIndices, {input[0].cols[0], input[1].cols[0]}}};
        tryAlignInputOutput(inputPaired, output);
        for(int col: util::i_range(2)){
            if(inputPaired[0].cols[col] != output[0].cols[col]){
                applyUnaryFunction(inputPaired, output, col, [](float in){return in;});
            }
        }
    }
};

class SplitVec2: public UnaryVec2, public Creatable<SplitVec2>{
public:
    SplitVec2(): UnaryVec2(){
        inputNames = {"vec2"};
        outputTypes = createFilledVec<FloatType, Type>(2);
        outputNames = {"x", "y"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // if different memory addresses, copy
        assert(input.size() == 1 && input[0].cols.size() == 2 && output.size() == 2 && output[0].cols.size() == 1 && output[1].cols.size() == 1);
        float_column_views outputPaired = {column_memory_view<float>{output[0].dimensionSizes, output[0].columnDimensionIndices, {output[0].cols[0], output[1].cols[0]}}};
        tryAlignInputOutput(input, outputPaired);
        for(int col: util::i_range(2)){
            if(input[0].cols[col] != outputPaired[0].cols[col]){
                applyUnaryFunction(input, outputPaired, col, [](float in){return in;});
            }
        }
        output[0].cols = {outputPaired[0].cols[0]};
        output[1].cols = {outputPaired[0].cols[1]};
    }
};

class Vec2Norm: public UnaryVec2, public Creatable<Vec2Norm>{
public:
    Vec2Norm(): UnaryVec2("", "len"){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyMultiDimUnaryFunction(input, output, {0,1}, [](const std::vector<float>& in){return std::sqrt(in[0] * in[0] + in[1] * in[1]);});
    }
};

class Vec2Square: public UnaryVec2, public Creatable<Vec2Square>{
    Vec2Square(): UnaryVec2("", "square"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return in * in;});
        applyUnaryFunction(input, output, 1, [](float in){return in * in;});
    }
};

class UnaryVec3: public Unary<Vec3Type>{
public:
    UnaryVec3(std::string_view header = "", std::string_view body = ""): Unary(header, body){};
};

class CreateVec3: public UnaryVec3, public Creatable<CreateVec3>{
public:
    CreateVec3(): UnaryVec3(){
        inputTypes = createFilledVec<FloatType, Type>(3);
        inputNames = {"x", "y", "z"};
        outputNames = {"vec3"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // if different memory addresses, copy
        assert(input.size() == 3 && input[0].cols.size() == 1 && input[1].cols.size() == 1 && output.size() == 1 && output[0].cols.size() == 3);
        const float_column_views inputPaired = {column_memory_view<float>{input[0].dimensionSizes, input[0].columnDimensionIndices, {input[0].cols[0], input[1].cols[0], input[2].cols[0]}}};
        tryAlignInputOutput(inputPaired, output);
        for(int col: util::i_range(3)){
            if(inputPaired[0].cols[col] != output[0].cols[col]){
                applyUnaryFunction(inputPaired, output, col, [](float in){return in;});
            }
        }
    }
};

class SplitVec3: public UnaryVec3, public Creatable<SplitVec3>{
public:
    SplitVec3(): UnaryVec3(){
        inputNames = {"vec3"};
        outputTypes = createFilledVec<FloatType, Type>(3);
        outputNames = {"x", "y", "z"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // if different memory addresses, copy
        assert(input.size() == 1 && input[0].cols.size() == 3 && output.size() == 3 && output[0].cols.size() == 1 && output[1].cols.size() == 1);
        float_column_views outputPaired = {column_memory_view<float>{output[0].dimensionSizes, output[0].columnDimensionIndices, {output[0].cols[0], output[1].cols[0], output[2].cols[0]}}};
        tryAlignInputOutput(input, outputPaired);
        for(int col: util::i_range(3)){
            if(input[0].cols[col] != outputPaired[0].cols[col]){
                applyUnaryFunction(input, outputPaired, col, [](float in){return in;});
            }
        }
        output[0].cols = {outputPaired[0].cols[0]};
        output[1].cols = {outputPaired[0].cols[1]};
        output[2].cols = {outputPaired[0].cols[2]};
    }
};

class Vec3Norm: public UnaryVec3, public Creatable<Vec3Norm>{
public:
    Vec3Norm(): UnaryVec3("", "len"){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyMultiDimUnaryFunction(input, output, {0,1,2}, [](const std::vector<float>& in){return std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);});
    }
};

class Vec3Square: public UnaryVec3, public Creatable<Vec3Square>{
    Vec3Square(): UnaryVec3("", "square"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return in * in;});
        applyUnaryFunction(input, output, 1, [](float in){return in * in;});
        applyUnaryFunction(input, output, 2, [](float in){return in * in;});
    }
};

class UnaryVec4: public Unary<Vec4Type>{
public:
    UnaryVec4(std::string_view header = "", std::string_view body = ""): Unary(header, body){};
};

class CreateVec4: public UnaryVec4, public Creatable<CreateVec4>{
public:
    CreateVec4(): UnaryVec4(){
        inputTypes = createFilledVec<FloatType, Type>(4);
        inputNames = {"x", "y", "z", "w"};
        outputNames = {"vec4"};
    }

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // if different memory addresses, copy
        assert(input.size() == 4 && input[0].cols.size() == 1 && input[1].cols.size() == 1 && output.size() == 1 && output[0].cols.size() == 4);
        const float_column_views inputPaired = {column_memory_view<float>{input[0].dimensionSizes, input[0].columnDimensionIndices, {input[0].cols[0], input[1].cols[0], input[2].cols[0], input[3].cols[0]}}};
        tryAlignInputOutput(inputPaired, output);
        for(int col: util::i_range(4)){
            if(inputPaired[0].cols[col] != output[0].cols[col]){
                applyUnaryFunction(inputPaired, output, col, [](float in){return in;});
            }
        }
    }
};

class SplitVec4: public UnaryVec4, public Creatable<SplitVec4>{
public:
    SplitVec4(): UnaryVec4(){
        inputNames = {"vec4"};
        outputTypes = createFilledVec<FloatType, Type>(4);
        outputNames = {"x", "y", "z", "w"};
    };

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        // if different memory addresses, copy
        assert(input.size() == 1 && input[0].cols.size() == 4 && output.size() == 4 && output[0].cols.size() == 1 && output[1].cols.size() == 1);
        float_column_views outputPaired = {column_memory_view<float>{output[0].dimensionSizes, output[0].columnDimensionIndices, {output[0].cols[0], output[1].cols[0], output[2].cols[0], output[3].cols[0]}}};
        tryAlignInputOutput(input, outputPaired);
        for(int col: util::i_range(4)){
            if(input[0].cols[col] != outputPaired[0].cols[col]){
                applyUnaryFunction(input, outputPaired, col, [](float in){return in;});
            }
        }
        output[0].cols = {outputPaired[0].cols[0]};
        output[1].cols = {outputPaired[0].cols[1]};
        output[2].cols = {outputPaired[0].cols[2]};
        output[3].cols = {outputPaired[0].cols[3]};
    }
};

class Vec4Norm: public UnaryVec4, public Creatable<Vec4Norm>{
public:
    Vec4Norm(): UnaryVec4("", "len"){outputTypes[0] = FloatType::create();};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyMultiDimUnaryFunction(input, output, {0,1,2,3}, [](const std::vector<float>& in){return std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);});
    }
};

class Vec4Square: public UnaryVec4, public Creatable<Vec4Square>{
    Vec4Square(): UnaryVec4("", "square"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyUnaryFunction(input, output, 0, [](float in){return in * in;});
        applyUnaryFunction(input, output, 1, [](float in){return in * in;});
        applyUnaryFunction(input, output, 2, [](float in){return in * in;});
        applyUnaryFunction(input, output, 3, [](float in){return in * in;});
    }
};

// ------------------------------------------------------------------------------------------
// binary nodes
// ------------------------------------------------------------------------------------------

template<class T>
class Binary: public Node{
public:
    Binary(std::string_view header = "", std::string_view body = ""):
        Node(createFilledVec<T, Type>(2), {std::string(), std::string()}, createFilledVec<T, Type>(1), {std::string()}, header, body){};
};

class Plus: public Binary<FloatType>, public Creatable<Plus>{
public:
    Plus(): Binary("", "+"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a + b;});
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min + input[1].cols_min_max[0].min, input[0].cols_min_max[0].max + input[0].cols_min_max[0].max}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::plus, input, output);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min + input[1].cols_min_max[0].min, input[0].cols_min_max[0].max + input[0].cols_min_max[0].max}};
    }
};

class Minus: public Binary<FloatType>, public Creatable<Minus>{
public:
    Minus(): Binary("", "-"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a - b;});
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min - input[1].cols_min_max[0].max, input[0].cols_min_max[0].max - input[0].cols_min_max[0].min}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::minus, input, output);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min - input[1].cols_min_max[0].max, input[0].cols_min_max[0].max - input[0].cols_min_max[0].min}};
    }
};

class Multiplication: public Binary<FloatType>, public Creatable<Multiplication>{
public:
    Multiplication(): Binary("", "*"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a * b;});
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min * input[1].cols_min_max[0].min, input[0].cols_min_max[0].max * input[0].cols_min_max[0].max}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::multiplication, input, output);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min * input[1].cols_min_max[0].min, input[0].cols_min_max[0].max * input[0].cols_min_max[0].max}};
    }
};

class Division: public Binary<FloatType>, public Creatable<Division>{
public:
    Division(): Binary("", "/"){};

    // TODO: proper  min_max

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a / b;});
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min / input[1].cols_min_max[0].max, input[0].cols_min_max[0].max / input[0].cols_min_max[0].min}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::division, input, output);
        output[0].cols_min_max = {min_max_t{input[0].cols_min_max[0].min / input[1].cols_min_max[0].max, input[0].cols_min_max[0].max / input[0].cols_min_max[0].min}};
    }
};

class Pow: public Binary<FloatType>, public Creatable<Pow>{
public:
    Pow(): Binary("", "pow"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return std::pow(a, b);});
        output[0].cols_min_max = {min_max_t{std::pow(input[0].cols_min_max[0].min, input[1].cols_min_max[0].max), std::pow(input[0].cols_min_max[0].max, input[0].cols_min_max[0].max)}};
    }
    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::pow, input, output);
    }
};

class Reduction: public Binary<FloatType>{
public:
    Reduction(std::string_view header): Binary(header){inputTypes[0] = IndexType::create(); inputNames[0] = "Group index";}
};

class Min_Reduction: public Reduction, public Creatable<Min_Reduction>{
public:
    static constexpr float init_val = std::numeric_limits<float>::max();
    Min_Reduction(): Reduction("Min") {}

    // input[0] has to contain the data values, input[1] has to contain the indices
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        applyUnaryReductionFunction(input, init_val, output[0], [](float a, float b){return std::min(a, b);});
        output[0].cols_min_max = input[1].cols_min_max;
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::min_red, input, output);
        init_values.emplace_back(init_value{output[0].cols[0].data(), init_val});
        output[0].cols_min_max = input[1].cols_min_max;
    }
};

class Max_Reduction: public Reduction, public Creatable<Max_Reduction>{
public:
    static constexpr float init_val = std::numeric_limits<float>::lowest();
    Max_Reduction(): Reduction("Max") {}

    // input[0] has to contain the data values, input[1] has to contain the indices
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        applyUnaryReductionFunction(input, init_val, output[0], [](float a, float b){return std::max(a, b);});
        output[0].cols_min_max = input[1].cols_min_max;
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::max_red, input, output);
        init_values.emplace_back(init_value{output[0].cols[0].data(), init_val});
        output[0].cols_min_max = input[1].cols_min_max;
    }
};

class Sum_Reduction: public Reduction, public Creatable<Sum_Reduction>{
public:
    static constexpr float init_val = 0.f;
    Sum_Reduction(): Reduction("Sum") {}

    // input[0] has to contain the data values, input[1] has to contain the indices
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        applyUnaryReductionFunction(input, init_val, output[0], [](float a, float b){return a + b;});
        output[0].cols_min_max = input[1].cols_min_max;
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::sum_red, input, output);
        init_values.emplace_back(init_value{output[0].cols[0].data(), init_val});
        output[0].cols_min_max = input[1].cols_min_max;
    }
};

class Mul_Reduction: public Reduction, public Creatable<Mul_Reduction>{
public:
    static constexpr float init_val = 1.f;
    Mul_Reduction(): Reduction("Multiplication") {}

    // input[0] has to contain the data values, input[1] has to contain the indices
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        applyUnaryReductionFunction(input, init_val, output[0], [](float a, float b){return a * b;});
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::mul_red, input, output);
        init_values.emplace_back(init_value{output[0].cols[0].data(), init_val});
    }
};

class Average_Reduction: public Reduction, public Creatable<Average_Reduction>{
public:
    static constexpr float init_val = 0.f;
    Average_Reduction(): Reduction("Average") {}

    // input[0] has to contain the data values, input[1] has to contain the indices
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        applyUnaryReductionFunction(input, init_val, output[0], [](float a, float b){return a + b;}, [](float a, size_t c){return double(a) / c;});
        output[0].cols_min_max = input[1].cols_min_max;
    }

    void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output, std::vector<init_value>& init_values) const override{
        add_operation(operations, op_codes::avg_red, input, output);
        init_values.emplace_back(init_value{output[0].cols[0].data(), init_val});
        output[0].cols_min_max = input[1].cols_min_max;
    }
};

class StdDev_Reduction: public Reduction, public Creatable<StdDev_Reduction>{
public:
    StdDev_Reduction(): Reduction("Standard Deviation") {}

    // input[0] has to contain the data values, input[1] has to contain the indices
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        auto average_f = [](float a, size_t c){return double(a) / c;};
        // use stddev = sqrt(expectation(vals^2) - expectation(vals)^2))
        // calc expectation(vals) and copy to temporary array
        applyUnaryReductionFunction(input, 0, output[0], [](float a, float b){return as<float>(double(a) + b);}, average_f);
        std::vector<float> expect(output[0].cols[0].begin(), output[0].cols[0].end());
        // calc expectation(vals^2)
        applyUnaryReductionFunction(input, 0, output[0], [](float a, float b){return as<float>(a + double(b) * b);}, average_f);
        for(size_t i: util::size_range(output[0].cols[0]))
            output[0].cols[0][i] = std::sqrt(output[0].cols[0][i] - expect[i] * expect[i]);

        output[0].cols_min_max = input[1].cols_min_max;
    }
};


class Derivative: public Binary<FloatType>, public Creatable<Derivative>{
    enum class difference_t: uint32_t{
        forward,
        backward,
        central,
        COUNT
    };
    const structures::enum_names<difference_t> difference_names{
        "forward",
        "backward",
        "central"
    };
public:
    Derivative(): Binary("Derivative", "") 
    {
        inputNames[0] = "h";
        inputTypes[0]->data()(0,0) = 1.f;
        input_elements[middle_input_id]["Dimension"] = dimension_selector;
        util::json::add_enum(input_elements[middle_input_id], "Difference Method", difference_names, difference_t::central);
        inplace_possible = false;
    }

    void applyOperationCpu(const float_column_views& input, float_column_views& output) const override{
        // at index 0 the width has to be entered
        std::vector<size_t> dimension_indices, dimension_indices_back;
        std::vector<size_t> dimension_indices_c;
        std::string dimension = input_elements[middle_input_id]["Dimension"]["selected_dim"].get<std::string>();
        auto dim_ptr = std::find(input[1].dimensionNames.begin(), input[1].dimensionNames.end(), dimension);
        if(dim_ptr == input[1].dimensionNames.end())    
            throw std::runtime_error{"Derivative::applyOperationCpu() Dimension " + dimension + " not available with current data input. Reselect dimension"};
        uint32_t dim{uint32_t(dim_ptr - input[1].dimensionNames.begin())};
        switch(util::json::get_enum_val<difference_t>(input_elements[middle_input_id]["Difference Method"])){
        case difference_t::forward:
            for(size_t i: util::size_range(input[1].cols[0])){
                dimension_indices_c = dimension_indices = input[1].columnIndexToDimensionIndices(i);
                if(dimension_indices[dim] < input[1].dimensionSizes[dim] - 1)
                    dimension_indices[dim]++;
                output[0].cols[0][i] = (input[1].atDimensionIndices(dimension_indices) - input[1].cols[0][i]) / input[0].atDimensionIndices(dimension_indices_c);
            }
            break;
        case difference_t::backward:
            for(size_t i: util::size_range(input[1].cols[0])){
                dimension_indices_c = dimension_indices = input[1].columnIndexToDimensionIndices(i);
                if(dimension_indices[dim] > 0)
                    dimension_indices[dim]--;
                output[0].cols[0][i] = (input[1].cols[0][i] - input[1].atDimensionIndices(dimension_indices)) / input[0].atDimensionIndices(dimension_indices_c);
            }
            break;
        case difference_t::central:
            for(size_t i: util::size_range(input[1].cols[0])){
                dimension_indices_c = dimension_indices_back = dimension_indices = input[1].columnIndexToDimensionIndices(i);
                if(dimension_indices[dim] < input[1].dimensionSizes[dim] - 1)
                    dimension_indices[dim]++;
                if(dimension_indices_back[dim] > 0)
                    dimension_indices_back[dim]--;
                output[0].cols[0][i] = (input[1].atDimensionIndices(dimension_indices) - input[1].atDimensionIndices(dimension_indices_back)) / (2 * input[0].atDimensionIndices(dimension_indices_c));
            }
            break;
        }
    }
};

class PlusVec2: public Binary<Vec2Type>, public Creatable<PlusVec2>{
public:
    PlusVec2(): Binary("", "+"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a + b;});
        applyBinaryFunction(input, output, 1, [](float a, float b){return a + b;});
    }
};

class MinusVec2: public Binary<Vec2Type>, public Creatable<MinusVec2>{
public:
    MinusVec2(): Binary("", "+"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a - b;});
        applyBinaryFunction(input, output, 1, [](float a, float b){return a - b;});
    }
};

class MultiplicationVec2: public Binary<Vec2Type>, public Creatable<MultiplicationVec2>{
public:
    MultiplicationVec2(): Binary("", "+"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a * b;});
        applyBinaryFunction(input, output, 1, [](float a, float b){return a * b;});
    }
};

class DivisionVec2: public Binary<Vec2Type>, public Creatable<DivisionVec2>{
public:
    DivisionVec2(): Binary("", "+"){};

    virtual void applyOperationCpu(const float_column_views& input ,float_column_views& output) const override{
        applyBinaryFunction(input, output, 0, [](float a, float b){return a / b;});
        applyBinaryFunction(input, output, 1, [](float a, float b){return a / b;});
    }
};
}
}