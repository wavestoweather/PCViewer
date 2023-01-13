#pragma once
#include <cassert>
#include <cinttypes>
#include <iterator>
#include <functional>
#include <optional>

namespace util{
// ranges with integer values
class i_range {
public:
    constexpr i_range(int64_t end): _begin(0), _end(end < 0 ? 0: end), _step(1){}; // single element constructor 
    constexpr i_range(long begin, long end, long step = 1):
     _begin(begin), _end(end), _step(step){
        assert(step != 0 && "step of 0 is invalid");
        if((begin > end && step > 0) || (begin < end && step < 0))
            _begin = _end;
    };

    class iterator {
        friend class i_range;
    public:
        long int operator *() const { return i_; }
        const iterator &operator ++() { i_ += _step; return *this; }
        iterator operator ++(int) { iterator copy(*this); i_ += _step; return copy; }    
        bool operator ==(const iterator &other) const { return i_ == other.i_; }
        bool operator !=(const iterator &other) const { return i_ != other.i_; } 
    protected:
        iterator(long int start, long step = 1) : i_ (start), _step(step) { }    
    private:
        long i_, _step;
    };  

    iterator begin() const { return iterator(_begin, _step); }
    iterator end() const { return iterator(_end); }
private:
    long _begin, _end;
    long _step;
};

class size_range{
public:
    template<class T>
    constexpr size_range(const T& sizeable): _end(sizeable.size()){};

    class iterator{
        friend class size_range;
    public:
        unsigned long operator*() const { return _i;}
        const iterator& operator++() { ++_i; return *this; }
        iterator operator++(int) { iterator copy(*this); ++_i; return copy; }
        bool operator==(const iterator& o) const {return _i == o._i;}
        bool operator!=(const iterator& o) const {return _i != o._i;}
    protected:
        iterator(unsigned long start): _i(start){}
    private:
        unsigned long _i;
    };

    iterator begin() const { return iterator(0); }
    iterator end() const { return iterator(_end); }
private:
    unsigned long _end;
};

class rev_size_range{
public:
    template<class T>
    rev_size_range(const T& sizeable): _begin(static_cast<int64_t>(sizeable.size()) - 1){};

    class iterator{
        friend class rev_size_range;
    public:
        int64_t operator*() const { return _i;}
        const iterator& operator++() { --_i; return *this; }
        iterator operator++(int) { iterator copy(*this); --_i; return copy; }
        bool operator==(const iterator& o) const {return _i == o._i;}
        bool operator!=(const iterator& o) const {return _i != o._i;}
    protected:
        iterator(int64_t start): _i(start){}
        iterator() = default;
    private:
        int64_t _i{-1};
    };

    iterator begin() const { return iterator(_begin); }
    iterator end() const { return iterator(); }
private:
    int64_t _begin;
};

template<typename T>
class rev_iter {
private:
    T& iterable_;
public:
    explicit rev_iter(T& iterable) : iterable_{iterable} {}
    auto begin() const { return std::rbegin(iterable_); }
    auto end() const { return std::rend(iterable_); }
};

template<typename T>
class first_iter {
    T& iterable_;
public:
    class iterator{
        friend class first_iter;
    public:
        using iter_type = decltype(std::begin(iterable_));
        using deref_iter_type = decltype(*std::begin(iterable_));
        std::pair<deref_iter_type&, bool> operator*() {return {*_i, _i == std::begin(iter_)};}
        const iterator& operator++() { ++_i; return *this;}
        iterator& operator++(int) {iterator copy(*this); ++_i; return copy;}
        bool operator==(const iterator& o) const{return _i == o._i;}
        bool operator!=(const iterator& o) const{return _i != o._i;}
    protected:
        iterator(const T& iterable, iter_type iter): iter_(iterable), _i(iter) {} 
    private:
        const T& iter_;
        iter_type _i;
    };

    explicit first_iter(T& iterable) : iterable_(iterable) {}
    iterator begin() const {return iterator(iterable_, std::begin(iterable_));}
    iterator end() const {return iterator(iterable_, std::end(iterable_));}
};

template<typename T>
class last_iter {
    T& iterable_;
public:
    class iterator{
        friend class last_iter;
    public:
        using iter_type = decltype(std::begin(iterable_));
        using deref_iter_type = decltype(*std::begin(iterable_));
        std::pair<deref_iter_type&, bool> operator*() {return {*_i, _i == --std::end(iter_)};}
        const iterator& operator++() { ++_i; return *this;}
        iterator& operator++(int) {iterator copy(*this); ++_i; return copy;}
        bool operator==(const iterator& o) const{return _i == o._i;}
        bool operator!=(const iterator& o) const{return _i != o._i;}
    protected:
        iterator(const T& iterable, iter_type iter): iter_(iterable), _i(iter) {} 
    private:
        const T& iter_;
        iter_type _i;
    };

    explicit last_iter(T& iterable) : iterable_(iterable) {}
    iterator begin() const {return iterator(iterable_, std::begin(iterable_));}
    iterator end() const {return iterator(iterable_, std::end(iterable_));}
};

enum class iterator_pos{
    first,
    last,
    between
};
template<typename T>
class pos_iter {
    T& iterable_;
public:
    class iterator{
        friend class pos_iter;
    public:
        using iter_type = decltype(std::begin(iterable_));
        using deref_iter_type = decltype(*std::begin(iterable_));
        std::pair<deref_iter_type&, iterator_pos> operator*() {return {*_i, _i == std::begin(iter_) ? iterator_pos::first : _i == --std::end(iter_) ? iterator_pos::last : iterator_pos::between};}
        const iterator& operator++() { ++_i; return *this;}
        iterator& operator++(int) {iterator copy(*this); ++_i; return copy;}
        bool operator==(const iterator& o) const{return _i == o._i;}
        bool operator!=(const iterator& o) const{return _i != o._i;}
    protected:
        iterator(const T& iterable, iter_type iter): iter_(iterable), _i(iter) {} 
    private:
        const T& iter_;
        iter_type _i;
    };

    explicit pos_iter(T& iterable) : iterable_(iterable) {}
    iterator begin() const {return iterator(iterable_, std::begin(iterable_));}
    iterator end() const {return iterator(iterable_, std::end(iterable_));}
};

template<typename T>
class enumerate {
    T& iterable_;
public:
    class iterator{
        friend class enumerate;
    public:
        using iter_type = decltype(std::begin(iterable_));
        using deref_iter_type = decltype(*std::begin(iterable_));
        std::pair<deref_iter_type&, size_t> operator*() {return {*_i, _i - iter_.begin()};}
        const iterator& operator++() { ++_i; return *this;}
        iterator& operator++(int) {iterator copy(*this); ++_i; return copy;}
        bool operator==(const iterator& o) const{return _i == o._i;}
        bool operator!=(const iterator& o) const{return _i != o._i;}
    protected:
        iterator(const T& iterable, iter_type iter): iter_(iterable), _i(iter) {} 
    private:
        const T& iter_;
        iter_type _i;
    };

    constexpr explicit enumerate(T& iterable) : iterable_(iterable) {}
    iterator begin() const {return iterator(iterable_, std::begin(iterable_));}
    iterator end() const {return iterator(iterable_, std::end(iterable_));}
};

template<typename T>
class const_enumerate {
    const T& iterable_;
public:
    class iterator{
        friend class const_enumerate;
    public:
        using iter_type = decltype(std::begin(iterable_));
        using deref_iter_type = decltype(*std::begin(iterable_));
        std::pair<deref_iter_type&, size_t> operator*() {return {*_i, _i - iter_.begin()};}
        const iterator& operator++() { ++_i; return *this;}
        iterator& operator++(int) {iterator copy(*this); ++_i; return copy;}
        bool operator==(const iterator& o) const{return _i == o._i;}
        bool operator!=(const iterator& o) const{return _i != o._i;}
    protected:
        iterator(const T& iterable, iter_type iter): iter_(iterable), _i(iter) {} 
    private:
        const T& iter_;
        iter_type _i;
    };

    constexpr explicit const_enumerate(const T& iterable) : iterable_(iterable) {}
    iterator begin() const {return iterator(iterable_, std::begin(iterable_));}
    iterator end() const {return iterator(iterable_, std::end(iterable_));}
};

// ---------------------------------------------------------------------------------------------------------------------------------
// reanges operators for easier range querying. 
// For example usage see test/ranges_test.cpp 
// ---------------------------------------------------------------------------------------------------------------------------------
template<typename T>
class contains{
public:
    constexpr contains(const T& e): _e(e) {}
    const T& _e;
};
template<typename T, typename U>
bool operator|(const T& range, const contains<U>& e){
    for(const auto& el: range)
        if(el == e._e)
            return true;
    return false;
}

template<typename T>
class contains_if{
public:
    std::function<bool(const T&)> _f;
    constexpr contains_if(std::function<bool(const T&)> f): _f(f) {}
};
template<typename T, typename U>
bool operator|(const T& range, const contains_if<U>& e){
    for(const auto& el: range)
        if(e._f(el))
            return true;
    return false;
}

template<typename T>
class try_find{
public:
    constexpr try_find(const T& e): e(e) {}
    const T& e;
};
template<typename T>
using optional_ref = std::optional<std::reference_wrapper<T>>;
template<typename T, typename U>
optional_ref<U> operator|(T& range, const try_find<U>& e){
    for(auto& el: range)
        if(el == e.e)
            return {std::reference_wrapper<U>{el}};
    return {};
}

template<typename T>
class try_find_if{
public:
    std::function<bool(const T&)> f;
    constexpr try_find_if(std::function<bool(const T&)> f): f(f) {}
};
template<typename T, typename U>
optional_ref<U> operator|(T& range, const try_find_if<U>& e){
    for(auto& el: range)
        if(e.f(el))
            return {std::reference_wrapper<U>{el}};
    return {};
}

template<typename T>
class try_pick_if{
public:
    std::function<bool(const T&)> f;
    constexpr try_pick_if(std::function<bool(const T&)> f): f(f) {}
};
template<typename T, typename U>
std::optional<U> operator|(const T& range, const try_pick_if<U>& e){
    for(auto& el: range)
        if(e.f(el))
            return {el};
    return {};
}

}

