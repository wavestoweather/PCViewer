#pragma once
#include <inttypes.h>

class DataInterface{
public:
    virtual float& operator()(uint32_t index, uint32_t column) = 0;
};