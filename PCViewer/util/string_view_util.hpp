#pragma once
#include <string_view>

namespace util{
inline bool getline(std::string_view& input, std::string_view& element, char delimiter = '\n'){
    if(input.empty()){
        element = {};
        return false;
    }
    
    size_t delimiter_pos = input.find(delimiter);
    size_t start = delimiter_pos + 1;
    if(delimiter_pos == std::string_view::npos){
        delimiter_pos = input.size();
        start = delimiter_pos;
    }
    element = input.substr(0, delimiter_pos);
    input = input.substr(start, input.size() - start);
    return true;
}

inline void trim_inplace(std::string_view& str){
    str = str.substr(str.find_first_not_of(" "));
    size_t back = str.size() - 1;
    while(str[back] == ' ')
        --back;
    str = str.substr(0, back + 1);
}

inline std::string_view trim(const std::string_view& str){
    std::string_view v = str;
    trim_inplace(v);
    return v;
}

struct slice{
    char c;
    constexpr slice(char c): c(c) {}
};
template<typename T = char>
class sliced_string{
    std::optional<std::string> storage;
    T                slice;
    std::string_view string;
public:
    class iterator{
        friend class sliced_string;
        std::string_view rest{};
        std::string_view cur{};
        T                slice{};
    public:
        std::string_view operator*() const{return cur;};
        std::string_view& operator*() {return cur;};
        const iterator& operator++() {getline(rest, cur, slice); return *this;}
        iterator operator++(int) {iterator copy(*this); getline(rest, cur, slice); return copy;}
        iterator operator+(int a) {iterator copy(*this); for(auto i: i_range(a)) ++copy; return copy;}
        
        bool operator==(const iterator& o) const {return rest == o.rest && cur == o.cur;}
        bool operator!=(const iterator& o) const {return rest != o.rest || cur != o.cur;}
        
        std::string_view get_rest() const {return rest;}
    protected:
        constexpr iterator() = default;
        constexpr iterator(std::string_view string, T slice): rest(string), slice(slice) {getline(rest, cur, slice);}
    };
    iterator begin() const {return iterator(string, slice);}
    iterator end() const {return iterator();}
    std::string_view operator[](size_t i){return *(begin() + i);}

    constexpr sliced_string(std::string_view data, T slice = {'\n'}): slice(slice), string(data){}
    constexpr sliced_string(std::string&& data, T slice = {'\n'}): storage(std::move(data)), slice(slice), string(storage.value()){}
};
static sliced_string<char> operator|(std::string_view string, slice s){
    return sliced_string<char>(string, s.c);
}
static sliced_string<char> operator|(const char* string, slice s){
    return sliced_string<char>(std::string_view(string), s.c);
}
static sliced_string<char> operator|(std::string&& string, slice s){
    return sliced_string<char>(std::move(string), s.c);
}

//struct occurrences{
//    std::string_view c;
//    constexpr occurrences(std::string_view c): c(c) {}
//};
//template<typename T = char>
//class occurrence_pos{
//    std::optional<std::string> storage;
//    std::string_view occ;
//    std::string_view string;
//public:
//    class iterator{
//        friend class sliced_string;
//        std::string_view string{};
//        std::string_view occ{};
//        size_t           pos{};
//    public:
//        std::string_view operator*() const{return cur;};
//        std::string_view& operator*() {return cur;};
//        const iterator& operator++() {getline(rest, cur, slice); return *this;}
//        iterator operator++(int) {iterator copy(*this); getline(rest, cur, slice); return copy;}
//        iterator operator+(int a) {iterator copy(*this); for(auto i: i_range(a)) ++copy; return copy;}
//        
//        bool operator==(const iterator& o) const {return rest == o.rest && cur == o.cur;}
//        bool operator!=(const iterator& o) const {return rest != o.rest || cur != o.cur;}
//    protected:
//        constexpr iterator() = default;
//        constexpr iterator(std::string_view string, T slice): rest(string), slice(slice) {getline(rest, cur, slice);}
//    };
//    iterator begin() const {return iterator(string, slice);}
//    iterator end() const {return iterator();}
//    std::string_view operator[](size_t i){return *(begin() + i);}
//
//    constexpr sliced_string(std::string_view data, T slice = {'\n'}): slice(slice), string(data){}
//    constexpr sliced_string(std::string&& data, T slice = {'\n'}): storage(std::move(data)), slice(slice), string(storage.value()){}
//};

}