#pragma once
#include <vector>
#include <inttypes.h>
#include "LassoBrush.hpp"
#include "range.hpp"

namespace brushing{
    struct AxisRange{
        uint32_t axis;
        float min;
        float max;
    };
    using RangeBrush = std::vector<AxisRange>;

    template<typename T>
    static bool inBrush(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes, const std::vector<T>& data, float eps = 0 /*maximum distance from data*/, bool andBrushes = false /*If true point has ot be in all ranges*/){
        bool a{rangeBrushes.empty()};
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

    template<typename T>
    static void updateIndexActivation(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes, const std::vector<std::vector<T>*>& data, std::vector<uint8_t>& activations, float eps = 0 /*maximum distance from data*/, bool andBrushes = false /*If true point has ot be in all ranges*/){
        for(size_t i: irange(*data[0])){
            bool a{rangeBrushes.empty()};
            for(auto& b: rangeBrushes){
                bool inRange{true};
                for(auto& r: b){
                    const uint32_t& ax = r.axis;
                    if((*data[ax])[i] + eps < r.min || (*data[ax])[i] - eps > r.max){
                        inRange = false;
                        break;
                    }
                }
                if(andBrushes && !inRange){
                    a = false;
                    break;
                }
                else if(inRange){
                    a = true;
                    break;
                }
            }
            if(!andBrushes && a){
                activations[i / 8] |= uint8_t(a) << i & 7;       //early out when data point is already accepted
                continue;
            }
            //TODO lasso selections

            if(andBrushes)
                activations[i / 8] &= uint8_t(a) << i & 7;
            else
                activations[i / 8] |= uint8_t(a) << i & 7;
        }
    }
}