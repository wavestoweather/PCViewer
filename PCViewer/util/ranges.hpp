#pragma once
#include <cassert>
#include <cinttypes>

namespace util{
// ranges with integer values
class i_range {
public:
    i_range(unsigned long end): _begin(0), _end(end), _step(1){}; // single element constructor 
    i_range(unsigned long begin, unsigned long end, long step = 1):
     _begin(begin), _end(end), _step(step){
        assert(step != 0 && ((begin > end && step < 0) || step > 0) && "Infinit loop detected");
        assert(((end - begin) % step == 0 || -(end - begin) % -step == 0) && "range results in an infinit loop");
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
    unsigned long _begin, _end;
    long _step;
};

class size_range{
public:
    template<class T>
    size_range(const T& sizeable): _end(sizeable.size()){};

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
    rev_size_range(const T& sizeable): _begin(sizeable.size() - 1){};

    class iterator{
        friend class rev_size_range;
    public:
        int64_t operator*() const { return _i;}
        const iterator& operator++() { --_i; return *this; }
        iterator operator++(int) { iterator copy(*this); --_i; return copy; }
        bool operator==(const iterator& o) const {return _i == o._i;}
        bool operator!=(const iterator& o) const {return _i != o._i;}
    protected:
        iterator(unsigned long start): _i(start){}
    private:
        int64_t _i;
    };

    iterator begin() const { return iterator(_begin); }
    iterator end() const { return iterator(-1); }
private:
    unsigned long _begin;
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
}
