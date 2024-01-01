#pragma once

#include <memory_view.hpp>
#include <thread>
#include <semaphore.hpp>
#include <atomic>
#include <string_view>
#include <functional>

namespace structures{
class priority_sorter{
public:
    // currently cpu only, so no gpu snychronization needed
    struct sorting_info{
        std::string_view                dl_id{};

        std::vector<std::atomic<bool>*> cpu_signal_flags{};
        std::vector<std::atomic<bool>*> cpu_unsignal_flags{};
    };

    priority_sorter();
    ~priority_sorter();

    void add_sort_task(const sorting_info& sorting_info);
    void wait_for_completion();
private:
    using unique_task = std::unique_ptr<sorting_info>;
    semaphore               _task_semaphore{};
    std::vector<unique_task>_sort_tasks{};
    std::mutex              _task_add_mutex{};
    std::thread             _task_thread{};
    std::atomic<bool>       _thread_finish{};
    std::atomic<uint32_t>   _wait_completion{};
    semaphore               _completion_semaphore{};

    void _task_thread_function();
};
}

namespace globals{
extern structures::priority_sorter priority_sorter;
}