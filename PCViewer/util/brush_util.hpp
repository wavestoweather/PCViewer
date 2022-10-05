#pragma once

#include <brushes.hpp>
#include <drawlists.hpp>
#include <array_struct.hpp>
#include <vk_util.hpp>
#include <vma_initializers.hpp>
#include <vma_util.hpp>
#include <brusher.hpp>
#include <vk_mem_alloc.h>

namespace util{
namespace brushes{
struct gpu_brush{

};

inline structures::dynamic_struct<gpu_brush, float> create_gpu_brush_data(const structures::range_brushes& range_brushes, const structures::lasso_brushes& lasso_brushes){
    return {};
}

// uploads changed local and global brushes
inline void upload_changed_brushes(){
    // global brushes
    structures::dynamic_struct<gpu_brush, float> global_brush_data;
    if(globals::global_brushes.changed){
        global_brush_data = create_gpu_brush_data(globals::global_brushes.read().ranges, globals::global_brushes.read().lassos);
        if(global_brush_data.byte_size() > util::vma::get_buffer_size(globals::global_brushes.read().brushes_gpu)){
            util::vk::destroy_buffer(globals::global_brushes().brushes_gpu);
            auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, global_brush_data.byte_size());
            auto alloc_info = util::vma::initializers::allocationCreateInfo();
            globals::global_brushes().brushes_gpu = util::vk::create_buffer(buffer_info, alloc_info);
        }
        structures::stager::staging_buffer_info staging_info{};
        staging_info.dst_buffer = globals::global_brushes.read().brushes_gpu.buffer;
        staging_info.common.data_upload = global_brush_data.data();
        globals::stager.add_staging_task(staging_info);
    }

    // local brushes
    std::vector<structures::dynamic_struct<gpu_brush, float>> local_brush_data;
    if(globals::drawlists.changed){
        for(const auto& [id, dl]: globals::drawlists.read()){
            if(!dl.changed || !dl.read().local_brushes.changed)
                continue;
            local_brush_data.push_back(create_gpu_brush_data(dl.read().local_brushes.read().ranges, dl.read().local_brushes.read().lassos));
            auto& brush_data = local_brush_data.back();

            if(brush_data.byte_size() > util::vma::get_buffer_size(dl.read().local_brushes.read().brushes_gpu)){
                util::vk::destroy_buffer(dl.read().local_brushes.read().brushes_gpu);
                auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, brush_data.byte_size());
                auto alloc_info = util::vma::initializers::allocationCreateInfo();
                globals::drawlists()[id]().local_brushes().brushes_gpu = util::vk::create_buffer(buffer_info, alloc_info);
            }

            structures::stager::staging_buffer_info staging_info{};
            staging_info.dst_buffer = dl.read().local_brushes.read().brushes_gpu.buffer;
            staging_info.common.data_upload = brush_data.data();
            globals::stager.add_staging_task(staging_info);
        }
    }

    globals::stager.wait_for_completion();
}

inline void update_drawlist_active_indices(){
    if(!globals::global_brushes.changed || !globals::drawlists.changed)
        return;
    
    for(const auto& [id, dl]: globals::drawlists.read()){
        if(!globals::global_brushes.changed || !dl.read().immune_to_global_brushes.changed || !dl.read().local_brushes.changed)
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
}   
}