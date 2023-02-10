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
#include "../util/json_util.hpp"

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
    ImVec4 color() const override{return {1.f,0.f,0.f,1.f};};
    column_memory_view<float> data() override {return column_memory_view(memory_view(_d));};
private:
    float _d;
};

class IndexType: public Type, public Creatable<IndexType, Type>{
    float _d{}; // default index is 0
public:
    ImVec4 color() const override{return {.1f,.1f,1.f,1.f};}
    column_memory_view<float> data() override {return column_memory_view(memory_view(_d));}
};

class ConstantFloatType: public Type, public Creatable<ConstantFloatType, Type>{
    ImVec4 color() const override{return {1.f, .5f, 0.f, 1.f};};
    column_memory_view<float> data() override{return {};};    // always returns null pointer as data is not changable
};

class VectorType: public Type,  public Creatable<VectorType, Type>{
public:
    ImVec4 color() const override{return {0.f,1.f,0.f,1.f};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::vector<float> _d;
};

class Vec2Type: public Type,  public Creatable<Vec2Type, Type>{
public:
    ImVec4 color() const override{return {.5f, .5f, .5f, 1.f};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::array<float, 2> _d;
};

class Vec3Type: public Type,  public Creatable<Vec3Type, Type>{
public:
    ImVec4 color() const override{return {1.f,1.f,1.f,1.f};};
    column_memory_view<float> data() override{return column_memory_view(memory_view(_d));};
private:
    std::array<float, 3> _d;
};

class Vec4Type: public Type,  public Creatable<Vec4Type, Type>{
public:
    ImVec4 color() const override{return {.1f, .1f, .1f, 1.f};};
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
    const crude_json::value dimension_selector = crude_json::object{{"type", "dim_sel"}, {"selected_dim", "Select"}};
    const crude_json::value drawlist_templatelist_selector = crude_json::object{{"type", "dl_tl_sel"}, {"selected_dl_tl", "Select"}};

    std::vector<std::unique_ptr<Type>> inputTypes;
    std::vector<std::string> inputNames;
    std::vector<std::unique_ptr<Type>> outputTypes;
    std::vector<std::string> outputNames;

    std::string name;
    std::string middleText;

    bool inplace_possible;

    crude_json::value input_elements;

    Node(std::vector<std::unique_ptr<Type>>&& inputTypes = {},
        std::vector<std::string>&& inputNames = {},
        std::vector<std::unique_ptr<Type>>&& outputTypes = {},
        std::vector<std::string>&& outputNames = {}, 
        std::string_view header = {}, std::string_view mt = {},
        bool inplace_possible = true):
        inputTypes(std::move(inputTypes)),
        inputNames(inputNames),
        outputTypes(std::move(outputTypes)),
        outputNames(outputNames),
        name(header),
        middleText(mt),
        inplace_possible(inplace_possible),
        input_elements(crude_json::type_t::object){}

    virtual int outputChannels() const { int count{}; for(const auto& t: outputTypes) count += static_cast<int>(t->data().cols.size()); return count;};
    virtual void applyOperationCpu(const float_column_views& input, float_column_views& output) const = 0;
    virtual void applyOperationGpu(std::stringstream& operations, const float_column_views& input, float_column_views& output) const{};
    virtual void imguiMiddleElements(const std::vector<std::string_view>& attributes = {}, const std::vector<std::string_view>& drawlists = {}, const std::vector<std::string_view>& templatelists = {}) { 
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
                if(util::json::is_dimension_selection(val)){
                    std::string dim = val["selected_dim"].get<std::string>();
                    if(!(attributes | util::contains(std::string_view(dim))))
                        val["selected_dim"] = std::string("Select");
                    if(ax::NodeEditor::BeginNodeCombo("Dimension", dim.c_str())){
                        for(auto a: attributes)
                            if(ImGui::MenuItem(a.data()))
                                val["selected_dim"] = std::string(a);
                        ax::NodeEditor::EndNodeCombo();
                    }
                }
                if(util::json::is_drawlist_templatelist_selection(val)){
                    std::string tl_dl = val["selected_dl_tl"].get<std::string>();
                    if(!(drawlists | util::contains(std::string_view(tl_dl))) && !(templatelists | util::contains(std::string_view(tl_dl))))
                        val["selected_dl_tl"] = std::string("Select");
                    if(ax::NodeEditor::BeginNodeCombo("Drawlist/Templatelist", tl_dl.c_str())){
                        for(auto dl: drawlists)
                            if(ImGui::MenuItem(dl.data()))
                                val["selected_dl_tl"] = std::string(dl);
                        ImGui::Separator();
                        for(auto tl: templatelists)
                            if(ImGui::MenuItem(tl.data()))
                                val["selected_dl_tl"] = std::string(tl);
                        ax::NodeEditor::EndNodeCombo();
                    }
                }
                if(util::json::is_enumeration(val)){
                    int ind = int(val["chosen"].get<double>());
                    if(ax::NodeEditor::BeginNodeCombo(name.c_str(), val["choices"][ind].get<std::string>().c_str())){
                        for(size_t i: util::size_range(val["choices"])){
                            if(ImGui::MenuItem(val["choices"][i].get<std::string>().c_str()))
                                val["chosen"] = double(i);
                        }
                        ax::NodeEditor::EndNodeCombo();
                    }
                }
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
    Registry(std::string name, std::function<std::unique_ptr<Node>()> createFunction) {std::replace(name.begin(), name.end(), '_', ' '); if(nodes.count(name) == 0) nodes[name] = {createFunction(), createFunction};};
};

// registers the nodes with a standard constructor
#define REGISTER(class) static Registry classReg_##class(#class , class::create<>);

}
}