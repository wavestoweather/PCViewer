#pragma once
#include <vk_context.hpp>
#include <memory_view.hpp>
#include <thread>
#include <array>
#include <semaphore.hpp>

namespace structures{

// stager uses a single staging buffer which is used in fifo mode for uploading data larger than the staging buffer
class stager{
public:
    struct staging_info{
        VkBuffer                            dst_buffer{};
        size_t                              dst_buffer_offset{};
        util::memory_view<const uint8_t>    data{};
        util::memory_view<VkSemaphore>      wait_semaphores{};
        util::memory_view<uint32_t>         wait_flags{};
        util::memory_view<VkSemaphore>      signal_semaphores{};
        semaphore*                          cpu_semaphore{};
    };

    // needed for task_thread dispatch
    void init();
    // needed for join with task_thread
    void cleanup();
    // checks state of task_thread and restarts if broken down
    void add_staging_task(const staging_info& stage_info);
    // set staging buffer size (defaults to 128 MB)
    void set_staging_buffer_size(size_t size);

    ~stager() {assert(!_task_thread.joinable() && "Call to cleanup() missing before destruction");}

private:
    semaphore               _task_semaphore{};
    std::vector<staging_info> _staging_tasks{};
    std::mutex              _task_add_mutex{};
    std::array<VkFence, 2>  _upload_fences{};
    std::thread             _task_thread{};
    std::atomic_bool        _thread_finish{};
    VkCommandPool           _command_pool{};
    std::array<VkCommandBuffer, 2>_command_buffers{};
    std::atomic_size_t      _staging_buffer_size{1u<<27}; // default size is 128 MB
    buffer_info             _staging_buffer{};
    void*                   _staging_buffer_mapped{};

    void _task_thread_function();
};
}

namespace globals{
extern structures::stager stager;
}