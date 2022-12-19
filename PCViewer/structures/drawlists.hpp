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
#include <histogram_registry.hpp>
#include <dataset_registry.hpp>

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
using median_iteration = enum_iteration<median_type>;

struct drawlist{
    std::string             id{};
    std::string             name{};
    std::string_view        parent_dataset{};
    std::string_view        parent_templatelist{};
    struct appearance{
        ImVec4      color{1,1,1,1};
        bool        show{true};
        bool        show_histogram{true};
    };
    change_tracker<appearance> appearance_drawlist{};
    std::vector<float>      brush_ratios_to_parent{};
    change_tracker<bool>    immune_to_global_brushes{};
    change_tracker<appearance> appearance_median{};
    change_tracker<median_type> median_typ{};  
    dynamic_bitset<uint32_t> active_indices_bitset{};                

    buffer_info             median_buffer{};                // linear array buffer containing the median values for all attributes
    buffer_info             active_indices_bitset_gpu{};
    buffer_info             priority_colors_gpu{};
    robin_hood::unordered_map<std::string, buffer_info> priority_indices{}; // optional gpu permutation buffer for sorted indices for priority rendering. For standard pcp the indexlist can be found under "standard"
    thread_safe_hist_reg    histogram_registry{};           // thread safety is guaranteed by thread_safe structure
    std::optional<dataset_registry::scoped_registrator_t> dataset_registrator{}; // only used when dataset uses gpu streaming
    tracked_brush           local_brushes{};
    buffer_info             local_brushes_gpu{};

    // optional data for certain rendering types
    struct delayed_ops_info_t{
        std::atomic<bool>   priority_rendering_requested{};
        std::atomic<bool>   priority_rendering_sorting_started{false};
        std::atomic<bool>   priority_sorting_done{true};
        std::atomic<bool>   delayed_ops_done{true};
    }                       delayed_ops;

    //TODO: add cluster and line bundles

    bool                        any_change() const          {return appearance_drawlist.changed || immune_to_global_brushes.changed || appearance_median.changed || median_typ.changed || local_brushes.changed;}
    void                        clear_change()              {appearance_drawlist.changed = false; immune_to_global_brushes.changed = false; appearance_median.changed = false; median_typ.changed = false; local_brushes.changed = false;}
    dataset&                    dataset_write() const       {return globals::datasets().at(parent_dataset)();}
    const structures::dataset&  dataset_read() const        {return globals::datasets.read().at(parent_dataset).read();} 
    //templatelist&               templatelist_write() const  {return *globals::datasets().at(parent_dataset)().templatelist_index[parent_templatelist];}
    const structures::templatelist& const_templatelist()const{return *globals::datasets.read().at(parent_dataset).read().templatelist_index.at(parent_templatelist);}
    void                        destroy_local_gpu_buffer()  {
        util::vk::destroy_buffer(median_buffer);
        util::vk::destroy_buffer(active_indices_bitset_gpu);
        util::vk::destroy_buffer(priority_colors_gpu);
        for(auto& [key, el]: priority_indices)
            util::vk::destroy_buffer(el);
        util::vk::destroy_buffer(local_brushes_gpu);
    }
};
using tracked_drawlist = unique_tracker<drawlist>;
using drawlists_t = change_tracker<std::map<std::string_view, tracked_drawlist>>;
}

namespace globals{
extern structures::drawlists_t drawlists;
extern std::vector<std::string_view> selected_drawlists;
extern std::set<std::string_view> drawlists_to_delete;
}

#define DECL_DRAWLIST_READ(dl_id)   const structures::drawlist& drawlist_read() const  {return globals::drawlists.read().at(dl_id).read();}
#define DECL_DRAWLIST_WRITE(dl_id)        structures::drawlist& drawlist_write() const {return globals::drawlists()[dl_id]();}