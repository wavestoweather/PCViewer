#pragma once
#include <vk_context.hpp>
#include <memory_view.hpp>
#include <thread>
#include <array>
#include <semaphore.hpp>
#include <atomic>
#include <functional>
#include <logger.hpp>

namespace structures{

// stager uses a single staging buffer which is used in fifo mode for uploading data larger than the staging buffer
class stager{
public:
    enum class transfer_direction{
        upload,
        download,
        COUNT
    };
    struct task_base{
        friend class stager;    // needed to grant stager access to task_executed
        transfer_direction                  transfer_dir{transfer_direction::upload};
        util::memory_view<const uint8_t>    data_upload{};
        util::memory_view<uint8_t>          data_download{};
        std::vector<VkSemaphore>            wait_semaphores{};
        std::vector<VkPipelineStageFlags>   wait_flags{};
        std::vector<VkSemaphore>            signal_semaphores{};
        semaphore*                          cpu_semaphore{};
        std::vector<std::atomic<bool>*>     cpu_signal_flags{};
        std::vector<std::atomic<bool>*>     cpu_unsignal_flags{};

        task_base() = default;
        task_base(transfer_direction transfer_dir, util::memory_view<const uint8_t> data_upload): transfer_dir(transfer_dir), data_upload(data_upload){}
        task_base(transfer_direction transfer_dir, util::memory_view<uint8_t> data_download): transfer_dir(transfer_dir), data_download(data_download){}
        virtual ~task_base(){
            if(!task_executed) 
                ::logger << logging::warning_prefix << " A data transfer task was not executed. This might lead to missing data on cpu or gpu." << logging::endl;
        }
 
    private:
        bool                                task_executed{};
    };
    struct staging_buffer_info: public task_base{
        VkBuffer                            dst_buffer{};
        size_t                              dst_buffer_offset{};

        staging_buffer_info() = default;
        staging_buffer_info(VkBuffer dst_buffer, size_t dst_buffer_offset, const task_base& task_b):
            dst_buffer{dst_buffer},
            dst_buffer_offset{dst_buffer_offset},
            task_base{task_b}{}
        staging_buffer_info(VkBuffer dst_buffer, size_t dst_buffer_offset, transfer_direction transfer_dir, util::memory_view<const uint8_t> data):
            dst_buffer{dst_buffer},
            dst_buffer_offset{dst_buffer_offset},
            task_base{transfer_dir, data}{}
    };
    struct staging_image_info: public task_base{
        VkImage                             dst_image{};
        VkImageLayout                       start_layout{};
        VkImageLayout                       end_layout{};   // after staging the image is in this image layout
        VkImageSubresourceLayers            subresource_layers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        uint32_t                            bytes_per_pixel{4};
        VkOffset3D                          image_offset{};
        VkExtent3D                          image_extent{1, 1, 1};
    };

    stager() {};

    // needed for task_thread dispatch
    void init();
    // interrupts current transfers and joins with the task_thread
    void cleanup();
    // add an upload task
    void add_staging_task(staging_buffer_info& stage_info);
    // add an upload task
    void add_staging_task(staging_image_info& stage_info);
    // set staging buffer size (defaults to 128 MB)
    void set_staging_buffer_size(size_t size);
    // waiting for completion of all upload tasks
    void wait_for_completion();

    ~stager() {assert(!_task_thread.joinable() && "Call to cleanup() missing before destruction");}

private:
    using unique_task = std::unique_ptr<task_base>;
    semaphore               _task_semaphore{};
    std::vector<unique_task> _staging_tasks{};
    std::mutex              _task_add_mutex{};
    std::array<VkFence, 2>  _task_fences{};
    int                     _fence_index{};
    std::thread             _task_thread{};
    std::atomic_bool        _thread_finish{};
    std::atomic_uint32_t    _wait_completion{};
    structures::semaphore   _completion_sempahore{};
    VkCommandPool           _command_pool{};
    std::array<VkCommandBuffer, 2>_command_buffers{};
    std::atomic_size_t      _staging_buffer_size{1u<<26}; // default size is 128 MB
    buffer_info             _staging_buffer{};
    uint8_t*                _staging_buffer_mapped{};

    void _task_thread_function();
};
}

namespace globals{
extern structures::stager stager;
}