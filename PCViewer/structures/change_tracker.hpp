#pragma once
#include <atomic>
#include <memory>

namespace structures{
template<class T>
class change_tracker{
public:
    change_tracker(const change_tracker<T>& o): changed(o.changed.load()), _obj(o._obj) {}
    change_tracker(change_tracker<T>&& o): changed(o.changed.load()), _obj(std::move(o._obj)) {}
    change_tracker& operator=(const change_tracker<T>& o) {_obj = o._obj; changed = o.changed.load(); return *this;}
    change_tracker& operator=(change_tracker<T>&& o) {_obj = std::move(o._obj); changed = o.changed.load(); return *this;}

    template<typename... Args>
    change_tracker(Args&&... args): _obj(args...){}

    std::atomic_bool changed{false};

    // constant (reading) data access
    inline const T& read() const {return _obj;}
    // quick non const (writing) data access
    inline T& operator()() {changed = true; return _obj;}
    // explicit non const (writing) data access
    inline T& write() {changed = true; return _obj;}
    // non const (writing) data access no change signal
    inline T& ref_no_track(){return _obj;}

    // write via equal operator
    inline change_tracker<T>& operator=(const T& o){changed = true; _obj = o; return *this;}

private:
    T _obj{};
};

// same as change_tracker but the storage is done in a unique ptr and dereferencing is made easier
template<class T>
class unique_tracker{
public:
    unique_tracker(unique_tracker<T>&& o): changed(o.changed.load()), _obj(std::move(o._obj)) {}
    unique_tracker& operator=(unique_tracker<T>&& o) {_obj = std::move(o._obj); changed = o.changed.load(); return *this;}

    template<typename... Args>
    unique_tracker(Args&&... args): _obj(std::make_unique<T>(args...)){};

    std::atomic_bool changed{false};

    // constant (reading) data access
    inline const T& read() const {return *_obj;}
    // non const (writing) data access
    inline T& operator()() {changed = true; return *_obj;}
    // explicit non const (writing) data access
    inline T& write() {changed = true; return *_obj;}
    // non const (writing) data access no change signal
    inline T& ref_no_track(){return *_obj;}

private:
    std::unique_ptr<T> _obj{};
};
}