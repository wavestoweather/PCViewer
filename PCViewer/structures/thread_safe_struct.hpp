#pragma once

#include <mutex>

namespace structures{
// creates a thread safe wrapper around a given struct where you can obtain an accessor to lock the struct and guarantee thread safe access
template<typename T>
class thread_safe{
    T                   _struct{};
    mutable std::mutex  _mutex{};

public:
    class access_t{
        T& _struct;
        std::scoped_lock<std::mutex> _lock;
    public:
        access_t(T& _struct, std::mutex& m): _struct(_struct), _lock(m) {}
        T* operator->() {return &_struct;}
        T& operator*() {return _struct;}
        access_t& operator=(const T& o) {_struct = o; return *this;}
        access_t& operator=(T&& o) {_struct = std::move(o); return *this;}
    };

    class const_access_t{
        const T& _struct;
        std::scoped_lock<std::mutex> _lock;
    public:
        const_access_t(const T& _struct, std::mutex& m): _struct(_struct), _lock(m) {}
        const T* operator->() const {return &_struct;}
        const T& operator*() const {return _struct;}
    };

    thread_safe(const thread_safe& o): _struct(o._struct) {}
    thread_safe(thread_safe&& o): _struct(std::move(o._struct)), _mutex(std::move(o._mutex)) {}
    thread_safe& operator=(const thread_safe& o) {_struct = o._struct; return *this;}
    thread_safe& operator=(thread_safe&& o) {_struct = std::move(o._struct); _mutex = std::move(o._mutex); return *this;}

    template<typename... Args>
    thread_safe(Args&&... args): _struct(args...) {}

    access_t access() {return access_t(_struct, _mutex);}
    const_access_t const_access() const {return const_access_t(_struct, _mutex);}
};
}