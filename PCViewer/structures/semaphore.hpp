#pragma once
#include <mutex>
#include <condition_variable>

namespace structures{
class semaphore{
    std::mutex _mutex;
    std::condition_variable _cv;
    unsigned long _count = 0;
public:
    void release(int n = 1){
        std::lock_guard<decltype(_mutex)> lock(_mutex);
        _count += n;
        for(int i = 0; i < n; ++i)
            _cv.notify_one();
    }
    void acquire(){
        std::unique_lock<decltype(_mutex)> lock(_mutex);
        while(!_count)
            _cv.wait(lock);
        --_count;
    }
    unsigned long peekCount(){return _count;};
};
}