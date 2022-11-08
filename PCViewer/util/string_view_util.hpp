#pragma once
#include <string_view>

namespace util{
inline bool getline(std::string_view& input, std::string_view& element, char delimiter = '\n'){
    if(input.empty())
        return false;
    
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
}