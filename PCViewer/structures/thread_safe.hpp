#pragma once
#include <mutex>
#include <thread>

namespace structures{
    template<class T>
    class thread_safe{
    public:
        class locked_view{
        public:
            scoped_lock(thread_safe& t): _lock(t){t._lockId = std::this_thread::get_id();};

        private:
            std::scoped_lock<std::mutex> _lock; 
        };

        // get read reference (automatic read lock if not yet locked)
        const T& read() const {if(_lockId != std::this_thread::get_id()) throw std::runtime_error{"thread_safe::read() "}};
        // get write reference. If this object was not locked a runtime error is thrown 
        T& write();

        // automatic read reference 
        const T& operator() const{return read();};
        T& operator(){return write();};

    private:
        T _obj;
        mutable std::mutex _mutex;
        mutable std::thread::id _lockId;
    };
}