#pragma once
#include <vk_context.hpp>
#include <memory_view.hpp>
#include <thread>

namespace structures{
class stager{
public:
    struct stage_info{
        structures::buffer_info             dst_buffer;
        util::memory_view<const uint8_t>    data;
        util::memory_view<VkSemaphore>      wait_semaphores;
        util::memory_view<VkSemaphore>      signal_semaphores;
    };

    // needed for task_thread dispatch
    void init();
    // needed for join with task_thread
    void cleanup();
    // checks state of task_thread and restarts if broken down
    void add_staging_task(const stage_info& stage_info);

    ~stager() {assert(!_task_thread.joinable() && "Call to cleanup() missing before destruction");}

private:
    std::vector<stage_info> _staging_tasks{};
    std::mutex              _task_add_mutex{};
    VkFence                 _upload_fence{};
    std::thread             _task_thread{};
    std::atomic_bool        _thread_finish{};
    VkCommandPool           _command_pool{};

    void _task_thread_function();
};
}

namespace globals{
extern structures::stager stager;
}