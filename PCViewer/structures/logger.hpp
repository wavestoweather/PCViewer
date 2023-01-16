#pragma once
#include <ostream>
#include <iostream>
#include <sstream>
#include <array>
#include <ranges.hpp>
#include <enum_names.hpp>

namespace logging{
struct endline{}; static endline        endl;
static std::string_view                 info_prefix{"[info]"};
static std::string_view                 warning_prefix{"[warning]"};
static std::string_view                 error_prefix{"[error]"};
static std::string_view                 vulkan_validation_prefix{"[vk validation]"};

enum class level{
    l_1,        // only errors
    l_2,        // + vulkan validatin
    l_3,        // + warnings
    l_4,        // + additional info
    l_5,        // + per frame info
    all,
    COUNT
};

const structures::enum_names<level> level_names{
    "only error", 
    "+ vk validation",
    "+ warnings", 
    "+ additional info", 
    "+ per frame info", 
    "all", 
};
}

namespace structures{
template<uint32_t buffered_lines = 40>
class logger{
    std::array<std::stringstream, buffered_lines>   _last_lines{};
    uint32_t                                        _write_head{};
    std::string                                     _buffered_lines{};
public:
    logger() = default;
    logger(const logger&) = delete;
    logger& operator=(const logger&) = delete;

    bool                        write_to_cout{true};
    bool                        prepare_full_lines_string{false};
    logging::level              logging_level{logging::level::l_4};
    bool                        scroll_bottom{};

    static constexpr uint32_t   buffer_size{buffered_lines};

    template<typename T>
    logger& operator<<(const T& o){
        _last_lines[_write_head] << o;
        
        if(write_to_cout)
            std::cout << o;

        return *this;
    }

    logger& operator<<(const logging::endline& o){
        scroll_bottom = true;
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

    bool even_write_head_pos() const{
        return _write_head % 2 == 0;
    }
};
}

extern structures::logger<20> logger;