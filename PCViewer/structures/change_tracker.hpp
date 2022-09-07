#pragma once
#include <atomic>
#include <memory>

namespace structures{
template<class T>
class change_tracker{
public:
    template<typename... Args>
    change_tracker(Args&&... args): _obj(args...){}

    std::atomic_bool changed{false};

    const T& const_ref() const {return _obj;}
    const T& operator()() const {return _obj;}

    T& ref() {changed = true; return _obj;}
    T* operator->() {changed = true; return &_obj;}
    T* operator*() {changed = true; return &_obj;}

    T& ref_no_track(){return _obj;}

private:
    T _obj{};
};

// same as change_tracker but the storage is done in a unique ptr and dereferencing is made easier
template<class T>
class unique_tracker{
public:
    template<typename... Args>
    unique_tracker(Args&&... args): _obj(std::make_unique<T>(args...)){};

    std::atomic_bool changed{false};

    const T& const_ref() const {return *_obj;}
    const T& operator()() const {return *_obj;}

    T& ref() {changed = true; return *_obj;}
    T* operator->() {changed = true; return _obj.get();}
    T* operator*() {changed = true; return _obj.get();}

    T& ref_no_track(){return _obj;}

private:
    std::unique_ptr<T> _obj{};
};
}