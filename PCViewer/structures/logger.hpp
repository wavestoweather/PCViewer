#pragma once
#include <ostream>
#include <iostream>
#include <sstream>
#include <array>
#include <ranges.hpp>
namespace structures{
template<uint32_t buffered_lines = 20>
class logger: public std::ostream{
    std::array<std::stringstream, buffered_lines>   _last_lines{};
    uint32_t                                        _write_head{};
    std::string                                     _buffered_lines{};
public:
    struct endline{};

    logger() = default;
    logger(const logger&) = delete;
    logger& operator=(const logger&) = delete;

    bool write_to_cout{true};
    bool prepare_full_lines_string{false};

    static constexpr uint32_t buffer_size{buffered_lines};

    template<typename T>
    logger& operator<<(const T& o){
        _last_lines[_write_head] << o;
        
        if(write_to_cout)
            std::cout << o;

        return *this;
    }

    logger& operator<<(const endline& o){
        _write_head = ++_write_head % buffered_lines;
        _last_lines[_write_head].str("");

        if(prepare_full_lines_string){
            std::stringstream res;
            for(int i: util::i_range(buffered_lines)){
                int cur_ind = (i + _write_head) % buffered_lines;
                if(!_last_lines[cur_ind].str().empty())
                    res << "\n" << _last_lines[i].str();
            }
            _buffered_lines = res.str();
        }

        if(write_to_cout)
            std::cout << std::endl;

        return *this;
    }

    std::string get_last_line(int n = 0){
        return _last_lines[(_write_head + buffered_lines - 1 - n) % buffered_lines].str();
    }

    const std::string& get_last_lines() const{
        return _buffered_lines;
    }
};
}

namespace logging{
static structures::logger<20>::endline  endl;
static std::string_view                 info_prefix{"[info]"};
static std::string_view                 warning_prefix{"[warning]"};
static std::string_view                 error_prefix{"[error]"};
}

extern structures::logger<20> logger;