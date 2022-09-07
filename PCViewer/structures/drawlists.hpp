#pragma once
#include <string>
#include <data.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <array>
#include <brushes.hpp>

namespace structures{

enum class median_type: uint32_t{
    none,
    arithmetic,
    synthetic,
    geometric,
    COUNT
};
static std::array<std::string_view, static_cast<size_t>(median_type::COUNT)> median_type_names{
    "none",
    "arithmetic",
    "synthetic",
    "geometric"
};

enum class alpha_mapping_type: uint32_t{
	multiplicative,
	bound01,
	const_alpha,
	alpha_adoption,
    COUNT
};
static std::array<std::string_view, static_cast<size_t>(alpha_mapping_type::COUNT)> alpha_mapping_type_names{
	"multiplicative",
	"bound01",
	"const_alpha",
	"alpha_adoption"
};

struct drawlist{
    std::string             id;
    std::string             name;
    std::string_view        parent_dataset;
    std::string_view        parent_templatelist;
    struct appearance{
        ImVec4      color;
        bool        show;
        bool        show_histogram;
    };
    change_tracker<appearance> appearance_drawlist;
    std::vector<float>      brush_ratios_to_parent;
    change_tracker<bool>    immune_to_global_brushes;
    change_tracker<appearance> appearance_median;
    median_type             median_typ;  
    std::vector<bool>       active_indices_bitmap;                

    buffer_info             index_buffer;
    buffer_info             median_buffer;              // linear array buffer containing the median values for all attributes
    buffer_info             active_indices_bitmap_gpu;
    tracked_brushes         local_brushes;
    alpha_mapping_type      alpha_mapping_typ;

    //TODO: add cluster and line bundles

    dataset&                    get_dataset()   const {return globals::datasets->at(parent_dataset).ref();}
    const structures::dataset&  const_dataset() const {return globals::datasets().at(parent_dataset)();} 
    templatelist&               get_templatelist() const {return *globals::datasets->at(parent_dataset)->templatelist_index[parent_templatelist];}
    const structures::templatelist& const_templatelist()const {return *globals::datasets().at(parent_dataset)().templatelist_index.at(parent_templatelist);}
};
using tracked_drawlist = unique_tracker<drawlist>;
using drawlists_t = change_tracker<std::vector<tracked_drawlist>>;
}

namespace globals{
extern structures::drawlists_t drawlists;
}