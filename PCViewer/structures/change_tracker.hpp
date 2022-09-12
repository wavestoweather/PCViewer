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

    // constant (reading) data access
    const T& read() const {return _obj;}
    // quick non const (writing) data access
    T& operator()() {changed = true; return _obj;}
    // explicit non const (writing) data access
    T& write() {changed = true; return _obj;}
    // non const (writing) data access no change signal
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

    // constant (reading) data access
    const T& read() const {return *_obj;}
    // non const (writing) data access
    T& operator()() {changed = true; return *_obj;}
    // explicit non const (writing) data access
    T& write() {changed = true; return *_obj;}
    // non const (writing) data access no change signal
    T& ref_no_track(){return *_obj;}

private:
    std::unique_ptr<T> _obj{};
};
}