#pragma once
#include <vector>
#include <inttypes.h>
#include "LassoBrush.hpp"

namespace brushing{
    struct AxisRange{
        uint32_t axis;
        float min;
        float max;
    };
    using RangeBrush = std::vector<AxisRange>;

    static bool inBrush(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes, const std::vector<float>& data, float eps = 0 /*maximum distance from data*/, bool andBrushes = false /*If true point has ot be in all ranges*/){
        bool a{false};
        for(auto& b: rangeBrushes){
            bool inRange{true};
            for(auto& r: b){
                const uint32_t& ax = r.axis;
                if(data[ax] + eps < r.min || data[ax] - eps > r.max){
                    inRange = false;
                    break;
                }
            }
            if(andBrushes && !inRange)
                return false;
            else if(inRange){
                a = true;
                break;
            }
        }
        if(!andBrushes && a)
            return a;       //early out when data point is already accepted
        //TODO lasso selections

        return a;
    }
}