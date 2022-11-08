#pragma once
#include <chrono>
#include <ostream>

namespace structures{
class scoped_watch{
    std::ostream&   _ostream;
    std::string     _name;
    std::chrono::high_resolution_clock::time_point _start;
public:
    scoped_watch(std::ostream& stream, std::string_view display_name):
        _ostream(stream),
        _name(display_name)
    {
        _start = std::chrono::high_resolution_clock::now();
    }
    ~scoped_watch(){
        auto end = std::chrono::high_resolution_clock::now();
        _ostream << "Stopwatch " << _name << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - _start).count() * 1e-3 << " ms" << std::endl;
    }
};
}