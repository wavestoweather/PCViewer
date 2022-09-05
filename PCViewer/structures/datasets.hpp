#pragma once
#include <attributes.hpp>
#include <change_tracker.hpp>
#include <data.hpp>
#include <memory>

namespace structures{

class dataset{
    std::string id;
    std::string display_name;
    std::string backing_data;   // filename or folder of the data on the hdd
    data<float> float_data;
    data<float> half_data;      // TODO: change to half
    
};

}

namespace globals{

using changing = structures::change_tracker;
using datasets_t = changing<std::map<std::string_view, std::unique_ptr<changing<structures::dataset>>>>;
static datasets_t& datasets(){
    static datasets_t d;    // is already copy restricted as unique_ptr is not copyable
    return d;
}

}