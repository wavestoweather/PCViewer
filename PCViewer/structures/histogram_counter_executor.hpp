#pragma once
#include <memory_view.hpp>
#include <thread>
#include <semaphore.hpp>
#include <atomic>
#include <string_view>
#include <gpu_timing.hpp>

namespace structures{
class histogram_counter{
public:
    struct histogram_count_info{
        std::string_view                    dl_id{};
        bool                                clear_counts{true};

        util::memory_view<VkSemaphore>      signal_semaphores{};
        semaphore*                          cpu_semaphore{};        // used to signal to the cpu that the gpu signal_semaphore has been used in a queue commit

        gpu_timing_info*                    count_timing_info{};
        gpu_timing_info*                    histograms_timing_info{};
    };

    // initialization via init() to avoid global initialization before vulkan context exists
    void init();
    // explicit cleanup to avoid destroying vulkan context while task is running
    void cleanup();

    void add_count_task(const histogram_count_info& count_info);

    void wait_for_completion();
private:
    using unique_task = std::unique_ptr<histogram_count_info>;
    semaphore               _task_semaphore{};
    std::vector<unique_task>_count_tasks{};
    std::mutex              _task_add_mutex{};
    std::thread             _task_thread{};
    std::atomic_bool        _thread_finish{};
    std::atomic_uint32_t    _wait_completion{};
    semaphore               _completion_semaphore{};
    VkSemaphore             _last_count_semaphore{};
    VkCommandPool           _wait_semaphore_pool{};
    VkCommandBuffer         _wait_semaphore_command{};
    VkFence                 _wait_semaphore_fence{};

    void _task_thread_function();
};
}

namespace globals{
extern structures::histogram_counter histogram_counter;
}