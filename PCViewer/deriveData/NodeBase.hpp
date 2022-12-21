#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <map>
#include <array>
#include "../imgui_nodes/imgui_node_editor.h"
#include "../imgui_nodes/utilities/widgets.h"
#include "../imgui_nodes/crude_json.h"
#include "../imgui/imgui_stdlib.h"
#include "MemoryView.hpp"

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


using float_column_views = std::vector<column_memory_view<float>>;
namespace Nodes{
class Node{
public:
    const std::string middle_input_id = "middle_inputs";
    const std::string input_input_id = "input_inputs";  // this should always be an array

    std::vector<std::unique_ptr<Type>> inputTypes;
    std::vector<std::string> inputNames;
    std::vector<std::unique_ptr<Type>> outputTypes;
    std::vector<std::string> outputNames;

    std::string name;
    std::string middleText;

    crude_json::value input_elements;

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
        middleText(mt),
        input_elements(crude_json::type_t::object){}

    virtual int outputChannels() const { uint32_t count{}; for(const auto& t: outputTypes) count += t->data().cols.size();return count;};
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const = 0;
    virtual void imguiMiddleElements() { 
        if(middleText.size()) ImGui::TextUnformatted(middleText.c_str());
        ImGui::PushItemWidth(70);
        if(input_elements.contains(middle_input_id)){
            for(auto& [name, val]: input_elements[middle_input_id].get<crude_json::object>()){
                if(val.is_string())
                    ImGui::InputText(name.c_str(), &val.get<std::string>());
                if(val.is_number())
                    ImGui::InputDouble(name.c_str(), &val.get<double>());
                if(val.is_boolean())
                    ImGui::Checkbox(name.c_str(), &val.get<bool>());
            }
        }
        ImGui::PopItemWidth();
    }
    virtual void imguiInputPinElement(int pin){
        ImGui::PushItemWidth(70);
        if(input_elements.contains(input_input_id)){
            assert(input_elements[input_input_id].is_array());
            ImGui::PushID(input_elements[input_input_id].get<crude_json::array>().data());
            if(input_elements[input_input_id][pin].is_number())
                ImGui::InputDouble("##d", &input_elements[input_input_id][pin].get<double>());
            ImGui::PopID();
        }
        if(inputNames[pin].size()){
            ImGui::SameLine();
            ImGui::Text("%s", inputNames[pin].data());
        }
        ImGui::PopItemWidth();
    }
};

struct Registry{
    struct Entry{
        std::unique_ptr<Node> prototype;
        std::function<std::unique_ptr<Node>()> create;
    };
    static std::map<std::string, Entry> nodes;
    Registry(std::string name, std::function<std::unique_ptr<Node>()> createFunction) {if(nodes.count(name) == 0) nodes[name] = {createFunction(), createFunction};};
};

// registers the nodes with a standard constructor
#define REGISTER(class) static Registry classReg_##class(#class , class::create<>);

}
}