#pragma once
#include <chrono>
#include <thread>

namespace structures{
struct frame_limiter{
    float                                       frame_time_micro;
    std::chrono::_V2::system_clock::time_point  last_time;

    frame_limiter(int fps = 60): frame_time_micro(1e6/fps), last_time(std::chrono::system_clock::now()) {}
    void end_frame(){
        auto end = std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - last_time).count();
        if(diff < frame_time_micro)
            std::this_thread::sleep_for(std::chrono::duration<float, std::micro>(frame_time_micro - float(diff)));
        last_time = end;
    }
};
}