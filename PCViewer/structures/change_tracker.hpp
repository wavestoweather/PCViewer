#pragma once
#include <atomic>

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

private:
    T _obj;
};
}