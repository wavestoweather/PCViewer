#pragma once
#include <chrono>
#include <robin_hood.h>
#include <vector>
#include <string>

namespace structures{
class stopwatch{
    using time_point = std::chrono::high_resolution_clock::time_point;
    robin_hood::unordered_map<std::string, std::vector<time_point>> _time_points;
public:
    // automatically restarts a coutning series if already available
    void start(const std::string& id = "d"){
        if(_time_points.contains(id))
            _time_points.erase(id);
        _time_points[id] = {std::chrono::high_resolution_clock::now()};
    }

    // adds a lap record and returns the time in ms since the last lap/start call
    double lap(const std::string& id = "d"){
        assert(_time_points.contains(id));
        _time_points[id].push_back(std::chrono::high_resolution_clock::now());
        return std::chrono::duration_cast<std::chrono::microseconds>(*(_time_points[id].end() - 1) - *(_time_points[id].end() - 2)).count() * 1e-3;
    }

    // returns the passed times in ms between the lap times specified by start and end index
    // values smaller than 0 for start and end index indicate place from the back (-1 is the last element)
    double lap_ms(const std::string& id = "d", int start_index = 0, int end_index = -1){
        assert(_time_points.contains(id));
        if(start_index < 0)
            start_index = _time_points[id].size() + start_index;
        if(end_index)
            end_index = _time_points[id].size() + end_index;
        assert(start_index >= 0 && start_index < _time_points[id].size());
        assert(end_index >= 0 && end_index < _time_points[id].size());
        assert(start_index < end_index);
        return std::chrono::duration_cast<std::chrono::microseconds>(*(_time_points[id].begin() + end_index) - *(_time_points[id].begin() + start_index)).count() * 1e-3;
    }
};
}