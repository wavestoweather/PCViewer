#pragma once
#include <memory_view.hpp>
#include <thread>
#include <semaphore.hpp>
#include <atomic>

namespace structures{
// class to load files from hdd in a separate thread
class file_loader{
public:
    struct load_info{
        std::string_view            src{};
        size_t                      src_offset{};
        util::memory_view<uint8_t>  dst{};
        semaphore*                  cpu_semaphore{};
    };

    void add_load_task(const load_info& load_info);
    // waiting for completion of all queued load tasks
    void wait_for_completion();

    file_loader();
    ~file_loader();

private:
    using unique_task = std::unique_ptr<load_info>;
    semaphore               _task_semaphore{};
    std::vector<unique_task>_loading_tasks{};
    std::mutex              _task_add_mutex{};
    std::thread             _task_thread{};
    std::atomic_bool        _thread_finish{};
    std::atomic_uint32_t    _wait_completion{};
    semaphore               _completion_semaphore{};

    void _task_thread_function();
};
}

namespace globals{
extern structures::file_loader file_loader;
}