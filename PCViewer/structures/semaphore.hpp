#pragma once
#include <mutex>
#include <condition_variable>
#include <memory>

namespace structures
{
    class semaphore
    {
        unsigned long _count{};
        std::mutex _mutex{};
        std::condition_variable _cv{};

    public:
        explicit semaphore() = default;
        ~semaphore()
        {
            _count = std::numeric_limits<int>::max();
            _cv.notify_all();
        }
        semaphore(const semaphore &o) = delete;
        semaphore &operator=(const semaphore &o) = delete;

        void release(int n = 1)
        {
            {
                std::lock_guard<decltype(_mutex)> lock(_mutex);
                _count += n;
            }
            _cv.notify_all();
        }
        void acquire()
        {
            std::unique_lock<decltype(_mutex)> lock(_mutex);
            _cv.wait(lock, [&]
                     { return (_count > 0); });
            assert(_count > 0);
            --_count;
        }
        bool try_acquire() {
            std::unique_lock<decltype(_mutex)> lock(_mutex);
            if (_count <= 0)
                return false;
            --_count;
            return true;
        }
        unsigned long peekCount() { return _count; };
    };
    using unique_semaphore = std::unique_ptr<semaphore>;
}