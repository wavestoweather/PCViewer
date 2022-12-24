#pragma once
#include <string_view>
#include <array>

namespace structures{

template<class T>
struct enum_names{
    using value_type = T;
    //enum_names(std::initializer_list<std::string_view> l): names(l){}
    typename std::array<std::string_view, static_cast<size_t>(T::COUNT)> names;
    std::string_view operator[](T place) const {return names[static_cast<size_t>(place)];}
    T operator[](std::string_view name) const {for(int i = 0; i < static_cast<int>(T::COUNT); ++i) if(names[i] == name) return static_cast<T>(i); return T::COUNT;}
};

template<class T>
struct enum_iteration{
    class iterator{
        friend struct enum_iteration;
    public:
        T operator*() const {return static_cast<T>(_i);}
        const iterator& operator++() {++_i; return *this;}
        iterator operator++(int) {iterator copy(*this); ++_i; return copy;}
        bool operator==(const iterator& o) const {return _i == o._i;}
        bool operator!=(const iterator& o) const {return _i != o._i;}
    protected:
        iterator(size_t start): _i(start){}
    private:
        size_t _i;
    };

    iterator begin() const {return iterator(0);}
    iterator end() const {return iterator(static_cast<size_t>(T::COUNT));}
};
}