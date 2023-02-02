#pragma once
#include <cassert>
#include <cinttypes>
#include <iterator>
#include <functional>
#include <optional>
#include <std_util.hpp>

namespace util{
// ranges with integer values
template<typename T>
class i_range {
public:
    constexpr i_range(T end): _begin(0), _end(end < 0 ? 0: end), _step(1){}; // single element constructor 
    constexpr i_range(T begin, T end, T step = T(1)):
     _begin(begin), _end((end - begin + step - std::sign(step)) / step * step + begin), _step(step){
        assert(step != 0 && "step of 0 is invalid");
        if((begin > end && step > 0) || (begin < end && step < 0))
            _begin = _end;
    };

    class iterator {
        friend class i_range;
    public:
        T operator *() const { return i_; }
        const iterator &operator ++() { i_ += _step; return *this; }
        iterator operator ++(int) { iterator copy(*this); i_ += _step; return copy; }    
        bool operator ==(const iterator &other) const { return i_ == other.i_; }
        bool operator !=(const iterator &other) const { return i_ != other.i_; } 
    protected:
        iterator(T start, T step = 1) : i_ (start), _step(step) { }    
    private:
        T i_, _step;
    };  

    iterator begin() const { return iterator(_begin, _step); }
    iterator end() const { return iterator(_end); }
private:
    T _begin, _end;
    T _step;
};

class size_range{
public:
    template<class T>
    constexpr size_range(const T& sizeable): _end(sizeable.size()){};

    class iterator{
        friend class size_range;
    public:
        uint64_t operator*() const { return _i;}
        const iterator& operator++() { ++_i; return *this; }
        iterator operator++(int) { iterator copy(*this); ++_i; return copy; }
        bool operator==(const iterator& o) const {return _i == o._i;}
        bool operator!=(const iterator& o) const {return _i != o._i;}
    protected:
        iterator(uint64_t start): _i(start){}
    private:
        uint64_t _i;
    };

    iterator begin() const { return iterator(0); }
    iterator end() const { return iterator(_end); }
private:
    uint64_t _end;
};

class rev_size_range{
public:
    template<class T>
    rev_size_range(const T& sizeable): _begin(sizeable.size() - 1){};

    class iterator{
        friend class rev_size_range;
    public:
        uint64_t operator*() const { return _i;}
        const iterator& operator++() { --_i; return *this; }
        iterator operator++(int) { iterator copy(*this); --_i; return copy; }
        bool operator==(const iterator& o) const {return _i == o._i;}
        bool operator!=(const iterator& o) const {return _i != o._i;}
    protected:
        iterator(uint64_t start): _i(start){}
        iterator() = default;
    private:
        uint64_t _i{uint64_t(-1)};
    };

    iterator begin() const { return iterator(_begin); }
    iterator end() const { return iterator(); }
private:
    uint64_t _begin;
};

template<typename T>
class rev_iter {
private:
    std::optional<T> _storage;
    T& iterable_;
public:
    explicit rev_iter(T& iterable) : iterable_{iterable} {}
    explicit rev_iter(T&& iterable) : _storage(std::move(iterable)), iterable_(*_storage) {}
    auto begin() const { return std::rbegin(iterable_); }
    auto end() const { return std::rend(iterable_); }
    size_t size() const { return iterable_.size(); }
};

template<typename T>
class first_iter {
    std::optional<T> _storage;
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
    explicit first_iter(T&& iterable) : _storage(std::move(iterable)), iterable_(*_storage) {}
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
        std::pair<deref_iter_type&, bool> operator*() {return {*_i, _i == std::end(iter_) - 1};}
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

struct iterator_pos{
    bool first: 1;
    bool last: 1;
    bool between: 1;
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
        std::pair<deref_iter_type&, iterator_pos> operator*() {return {*_i, {_i == std::begin(iter_), _i == --std::end(iter_), _i != std::begin(iter_) && _i != --std::end(iter_)}};}
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
    std::optional<T> _storage;
    T& iterable_;
public:
    class iterator{
        friend class enumerate;
    public:
        using iter_type = decltype(std::begin(iterable_));
        using deref_iter_type = decltype(*std::begin(iterable_));
        std::pair<deref_iter_type&, size_t> operator*() {return {*_i, _iter_pos};}
        const iterator& operator++() { ++_i; ++_iter_pos; return *this;}
        iterator& operator++(int) {iterator copy(*this); ++_i; ++_iter_pos; return copy;}
        bool operator==(const iterator& o) const{return _i == o._i;}
        bool operator!=(const iterator& o) const{return _i != o._i;}
    protected:
        iterator(size_t iter_pos, iter_type iter): _iter_pos(iter_pos), _i(iter) {} 
    private:
        size_t _iter_pos;
        iter_type _i;
    };

    constexpr explicit enumerate(T& iterable) : iterable_(iterable) {}
    constexpr explicit enumerate(T&& iterable) : _storage(std::move(iterable)), iterable_(*_storage) {}
    iterator begin() const {return iterator(0, std::begin(iterable_));}
    iterator end() const {return iterator(iterable_.size(), std::end(iterable_));}
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
            return {std::reference_wrapper<U>(el)};
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

constexpr size_t n_pos{size_t(-1)};
template<typename T>
class index_of{
    std::optional<T> _storage;
public:
    const T& e;
    constexpr index_of(const T& e): e(e) {}
    constexpr index_of(T&& e): _storage(std::move(e)), e(*_storage) {}
};
template<typename T>
size_t operator|(const T& range, const index_of<T>& e){
    for(auto&& [el, i]: enumerate(range))
        if(e.e == el)
            return i;
    return n_pos;
}

template<typename T>
class index_of_if{
public:
    std::function<bool(const T&)> f;
    constexpr index_of_if(std::function<bool(const T&)> f): f(f) {}
};
template<typename T, typename U>
size_t operator|(const T& range, const index_of_if<U>& e){
    for(auto&& [el, i]: enumerate(range))
        if(e.f(el))
            return i;
    return n_pos;
}

class max{
};
template<typename T>
decltype(*std::declval<T>().begin()) operator|(T range, const max&){
    using return_type = decltype(*std::declval<T>().begin());
    return_type m{};
    for(auto&& [e, first]: first_iter(range)){
        if(first) m = e;
        else m = std::max(m, e);
    }
    return e;
}

class min{
};
template<typename T>
decltype(*std::declval<T>().begin()) operator|(T range, const min&){
    using return_type = decltype(*std::declval<T>().begin());
    return_type m{};
    for(auto&& [e, first]: first_iter(range)){
        if(first) m = e;
        else m = std::min(m, e);
    }
    return e;
}

template<typename T, typename = void>
struct is_resizable: std::false_type{};
template<typename T>
struct is_resizable<T, decltype(std::declval<T>().resize(0), void())> : std::true_type {};
template<typename T, typename = void>
struct is_sizeable: std::false_type{};
template<typename T>
struct is_sizeable<T, decltype(std::declval<T>().size(0), void())> : std::true_type {};
template<typename T>
class to{
};
template<typename T, typename U>
U operator|(const T& range, const to<U>& to){
    U ret;
    if constexpr(is_resizable<U>::value && is_sizeable<T>::value)
        ret.resize(range.size());
    for(auto&& e: range)
        ret.emplace_back(e);
    return ret;
}
}

