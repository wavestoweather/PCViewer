#pragma once

#include "DataInterface.hpp"
#include <vector>
#include <inttypes.h>

class CompressedData: public DataInterface{
public:
    std::vector<uint8_t> compressedData;
    float& operator()(uint32_t index, uint32_t column){
        return dummy;
    }
private:
    float dummy;
};