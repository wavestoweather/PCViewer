#pragma once
#include <string>
#include <data.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <brushes.hpp>
#include <vk_context.hpp>
#include <datasets.hpp>
#include <enum_names.hpp>
#include <dynamic_bitset.hpp>

namespace structures{

enum class median_type: uint32_t{
    none,
    arithmetic,
    synthetic,
    geometric,
    COUNT
};
static enum_names<median_type> median_type_names{
    "none",
    "arithmetic",
    "synthetic",
    "geometric"
};

struct drawlist{
    std::string             id{};
    std::string             name{};
    std::string_view        parent_dataset{};
    std::string_view        parent_templatelist{};
    struct appearance{
        ImVec4      color{1,1,1,1};
        bool        show{true};
        bool        show_histogram{};
    };
    change_tracker<appearance> appearance_drawlist{};
    std::vector<float>      brush_ratios_to_parent{};
    change_tracker<bool>    immune_to_global_brushes{};
    change_tracker<appearance> appearance_median{};
    change_tracker<median_type> median_typ{};  
    dynamic_bitset<uint32_t> active_indices_bitset{};                

    buffer_info             index_buffer{};
    buffer_info             median_buffer{};                // linear array buffer containing the median values for all attributes
    buffer_info             active_indices_bitset_gpu{};
    std::vector<buffer_info> derived_data_infos{};          // vulkan buffer need e.g. for large vis counting
    tracked_brushes         local_brushes{};

    //TODO: add cluster and line bundles

    dataset&                    dataset_write() const       {return globals::datasets().at(parent_dataset)();}
    const structures::dataset&  dataset_read() const        {return globals::datasets.read().at(parent_dataset).read();} 
    //templatelist&               templatelist_write() const  {return *globals::datasets().at(parent_dataset)().templatelist_index[parent_templatelist];}
    const structures::templatelist& const_templatelist()const {return *globals::datasets().at(parent_dataset)().templatelist_index[parent_templatelist];}
};
using tracked_drawlist = unique_tracker<drawlist>;
using drawlists_t = change_tracker<std::map<std::string_view, tracked_drawlist>>;
}

namespace globals{
extern structures::drawlists_t drawlists;
extern std::vector<std::string_view> selected_drawlists;
}