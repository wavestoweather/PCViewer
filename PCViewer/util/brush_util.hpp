#pragma once

#include <brushes.hpp>
#include <drawlists.hpp>
#include <array_struct.hpp>
#include <vk_util.hpp>
#include <vma_initializers.hpp>
#include <vma_util.hpp>
#include <brusher.hpp>
#include <vk_mem_alloc.h>
#include <stager.hpp>
#include <robin_hood.h>

namespace util{
namespace brushes{
struct gpu_brush{
    uint32_t range_brush_count;
    uint32_t lasso_brush_count;
    uint32_t lasso_brush_offset;
    uint32_t _;
};
using range_brush_refs = std::vector<const structures::range_brush*>;
using lasso_brush_refs = std::vector<const structures::lasso_brush*>;

inline structures::dynamic_struct<gpu_brush, float> create_gpu_brush_data(const range_brush_refs& range_brushes, const lasso_brush_refs& lasso_brushes){
    // creating an attribute major storage for the range_brushes to be able to convert the data more easily
    std::vector<std::map<uint32_t, std::vector<structures::min_max<float>>>> axis_brushes(range_brushes.size());
    uint32_t dynamic_size{static_cast<uint32_t>(range_brushes.size())};   // brush_offsets 
    for(int i: size_range(range_brushes)){
        auto& b = axis_brushes[i];
        for(const auto& range: *range_brushes[i])
            b[range.axis].push_back({range.min, range.max});
        dynamic_size += 1;                          // nAxisMaps
        dynamic_size += b.size();                   // axisOffsets
        for(const auto& [axis, ranges]: b){
            dynamic_size += 1;                      // nRanges
            dynamic_size += 1;                      // axis
            dynamic_size += ranges.size() * 2;      // ranges
        }
    }
    dynamic_size += lasso_brushes.size();           // brush_offsets
    for(int i: size_range(lasso_brushes)){
        const auto& lasso_brush = *lasso_brushes[i];
        dynamic_size += 1;                          // nPolygons
        dynamic_size += lasso_brush.size();         // polygonOffsets
        for(const auto& polygon: lasso_brush){
            dynamic_size += 2;                      // attr1, attr2
            dynamic_size += 1;                      // nBorderPoints
            dynamic_size += polygon.borderPoints.size() * 2; // borderPoints
        }
    }

    // converting teh brush data to a linearized array
    // priority for linearising trhe brush data: brushes, axismap, ranges
    // the float array following the brushing info has the following layout
    //      vector<float> brushOffsets, vector<Brush> brushes;                                  // where brush offests describe the index in the float array from which the brush at index i is positioned
    //      with Brush = {flaot nAxisMaps, vector<float> axisOffsets, vector<AxisMap> axisMaps} // axisOffsets same as brushOffsetsf for the axisMap
    //      with AxisMap = {float nrRanges, fl_brushEventoat axis, vector<Range> ranges}
    //      with Range = {float min, float max}
    structures::dynamic_struct<gpu_brush, float> gpu_data(dynamic_size);
    gpu_data->range_brush_count = axis_brushes.size();
    uint32_t cur_offset{static_cast<uint32_t>(axis_brushes.size())};   // base offset for the first brush comes after the offsets
    for(int brush: size_range(axis_brushes)){
        gpu_data[brush] = cur_offset;           // brush_offset
        gpu_data[cur_offset++] = axis_brushes[brush].size();    // nAxisMaps
        uint32_t axis_offsets = cur_offset;
        cur_offset += axis_brushes[brush].size();               // skipping the axis offsets
        for(const auto& [axis, ranges]: axis_brushes[brush]){
            gpu_data[axis_offsets++] = cur_offset;              // axisOffsets
            gpu_data[cur_offset++] = ranges.size();             // nRanges
            gpu_data[cur_offset++] = axis;                      // axis
            for(const auto& range: ranges){
                gpu_data[cur_offset++] = range.min;             // min
                gpu_data[cur_offset++] = range.max;             // max
            }
        }
    }

    // converting the lasso data to a linearized array and appending after the range bruhes
    // priority for linearising the data: brushes, lassos, border_points
    // the float array following the range_brushes has the following layout
    //      vector<float> brush_offsets, vector<lasso_brush> brushes;           // where offsets are the absolute offsets of the lassos, offsets begin at gpu_brush.lasso_brush_offset
    //      with lasso_brush = {float nPolygons, vector<float> polygonOffsets, vector<polygon> polygons}
    //      with polygon = {float attr1, float attr2, float nBorderPoints, vector<vec2> borderPoints}
    //      with vec2 = {float p1, float p2}
    gpu_data->lasso_brush_count = lasso_brushes.size();
    gpu_data->lasso_brush_offset = cur_offset;
    cur_offset += lasso_brushes.size();
    for(int i: size_range(lasso_brushes)){
        const auto& lasso_brush = *lasso_brushes[i];
        gpu_data[gpu_data->lasso_brush_offset + i] = cur_offset;    // brush_offset
        gpu_data[cur_offset++] = lasso_brush.size();                // nPolygons
        uint32_t polygon_offsets = cur_offset;
        cur_offset += lasso_brush.size();                           // skipping the polygon_offsets
        for(const auto& polygon: lasso_brush){
            gpu_data[polygon_offsets++] = cur_offset;               // polygon_offsets
            gpu_data[cur_offset++] = polygon.attr1;                 // attr1
            gpu_data[cur_offset++] = polygon.attr2;                 // attr2
            gpu_data[cur_offset++] = polygon.borderPoints.size();   // nBorderPoints
            for(const auto& b: polygon.borderPoints){
                gpu_data[cur_offset++] = b.x;                       // p1
                gpu_data[cur_offset++] = b.y;                       // p2
            }
        }
    }

    return std::move(gpu_data);
}

// uploads changed local and global brushes
inline void upload_changed_brushes(){
    // global brushes
    structures::dynamic_struct<gpu_brush, float> global_brush_data;
    bool wait_stager = false;
    if(globals::global_brushes.changed){
        range_brush_refs range_brushes;
        lasso_brush_refs lasso_brushes;
        for(const auto& brush: globals::global_brushes.read()){
            range_brushes.push_back(&brush.read().ranges);
            lasso_brushes.push_back(&brush.read().lassos);
        }
        global_brush_data = create_gpu_brush_data(range_brushes, lasso_brushes);
        if(global_brush_data.byte_size() > globals::global_brushes.brushes_gpu.size){
            util::vk::destroy_buffer(globals::global_brushes.brushes_gpu);
            auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, global_brush_data.byte_size());
            auto alloc_info = util::vma::initializers::allocationCreateInfo();
            globals::global_brushes.brushes_gpu = util::vk::create_buffer(buffer_info, alloc_info);
        }
        structures::stager::staging_buffer_info staging_info{};
        staging_info.dst_buffer = globals::global_brushes.brushes_gpu.buffer;
        staging_info.common.data_upload = global_brush_data.data();
        globals::stager.add_staging_task(staging_info);
        wait_stager = true;
    }

    // local brushes
    std::vector<structures::dynamic_struct<gpu_brush, float>> local_brush_data;
    if(globals::drawlists.changed){
        for(const auto& [id, dl]: globals::drawlists.read()){
            if(!dl.changed || !dl.read().local_brushes.changed)
                continue;
            range_brush_refs range_brushes{&dl.read().local_brushes.read().ranges};
            lasso_brush_refs lasso_brushes{&dl.read().local_brushes.read().lassos};
            local_brush_data.push_back(create_gpu_brush_data(range_brushes, lasso_brushes));
            auto& brush_data = local_brush_data.back();

            if(brush_data.byte_size() > dl.read().local_brushes_gpu.size){
                util::vk::destroy_buffer(dl.read().local_brushes_gpu);
                auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, brush_data.byte_size());
                auto alloc_info = util::vma::initializers::allocationCreateInfo();
                globals::drawlists()[id]().local_brushes_gpu = util::vk::create_buffer(buffer_info, alloc_info);
            }

            structures::stager::staging_buffer_info staging_info{};
            staging_info.dst_buffer = dl.read().local_brushes_gpu.buffer;
            staging_info.common.data_upload = brush_data.data();
            globals::stager.add_staging_task(staging_info);
            wait_stager = true;
        }
    }

    if(wait_stager)
        globals::stager.wait_for_completion();
}

inline void update_drawlist_active_indices(){
    if(!globals::global_brushes.changed && !globals::drawlists.changed)
        return;
    
    for(const auto& [id, dl]: globals::drawlists.read()){
        if(!globals::global_brushes.changed && !dl.read().immune_to_global_brushes.changed && !dl.read().local_brushes.changed)
            continue;

        if(globals::global_brushes.changed && !dl.read().local_brushes.changed && dl.read().immune_to_global_brushes.read() && !dl.read().immune_to_global_brushes.changed)
            continue;

        pipelines::brusher::brush_info brush_info{};
        brush_info.drawlist_id = id;
        pipelines::brusher::instance().brush(brush_info);

        globals::drawlists()[id]().local_brushes.changed = false;
        globals::drawlists()[id]().immune_to_global_brushes.changed = false;
    }

    globals::global_brushes.changed = false;
}

inline const structures::range_brush& get_selected_range_brush_const(){
    switch(globals::brush_edit_data.brush_type){
    case structures::brush_edit_data::brush_type::global:
        assert(std::count_if(globals::global_brushes.read().begin(), globals::global_brushes.read().end(), [&](const structures::tracked_brush& b){return b.read().id == globals::brush_edit_data.global_brush_id;}) != 0);
        return std::find_if(globals::global_brushes.read().begin(), globals::global_brushes.read().end(), [&](const structures::tracked_brush& b){return b.read().id == globals::brush_edit_data.global_brush_id;})->read().ranges;
    case structures::brush_edit_data::brush_type::local:
        return globals::drawlists.read().at(globals::brush_edit_data.local_brush_id).read().local_brushes.read().ranges;
    default:
        assert(false && "Not yet implementd");
        return {};
    }
}

inline structures::range_brush& get_selected_range_brush(){
    switch(globals::brush_edit_data.brush_type){
    case structures::brush_edit_data::brush_type::global:
        assert(std::count_if(globals::global_brushes.read().begin(), globals::global_brushes.read().end(), [&](const structures::tracked_brush& b){return b.read().id == globals::brush_edit_data.global_brush_id;}) != 0);
        return std::find_if(globals::global_brushes().begin(), globals::global_brushes().end(), [&](const structures::tracked_brush& b){return b.read().id == globals::brush_edit_data.global_brush_id;})->write().ranges;
    case structures::brush_edit_data::brush_type::local:
        return globals::drawlists().at(globals::brush_edit_data.local_brush_id)().local_brushes().ranges;
    default:
        assert(false && "Not yet implementd");
    }
}

inline void delete_brushes(const robin_hood::unordered_set<structures::range_id>& brush_delete){
    auto& ranges = util::brushes::get_selected_range_brush();
    for(structures::range_id range: brush_delete){
        ranges.erase(std::find_if(ranges.begin(), ranges.end(), [&](const structures::axis_range& r){return r.id == range;}));
    }
    globals::brush_edit_data.selected_ranges.clear();
}
}   
}