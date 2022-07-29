#pragma once
#include <inttypes.h>

struct PriorityInfo{
    int axis{-1};       // -1 indicates that no priority should be queried
    float axisValue{};
};