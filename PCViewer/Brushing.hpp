#pragma once
#include <vector>
#include <thread>
#include <inttypes.h>
#include <map>
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
        // Range brushes ------------------------------------------------------
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

        // Lasso brushes ------------------------------------------------------
        for(const auto& lasso: lassoBrushes){
            int attr1 = lasso.attr1;
            int attr2 = lasso.attr2;
            bool inLasso = false;
            ImVec2 d{static_cast<float>(data[attr1]), static_cast<float>(data[attr2])};
            for(int i: irange(lasso.borderPoints)){
                const ImVec2& a = lasso.borderPoints[i];
                const ImVec2& b = lasso.borderPoints[(i + 1) % lasso.borderPoints.size()];
                // calculate line intersection with horizontal line, code from https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
                if( ((a.y > d.y) != (b.y > d.y)) &&
				    (d.x < (b.x - a.x) * (d.y - a.y) / (b.y - a.y) + a.x) )
				    inLasso = !inLasso;
            }
            a = a && inLasso;
            if(!a)
                break;
        }

        return a;
    }

    template<typename T>
    static void updateIndexActivation(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes, const std::vector<const std::vector<T>*>& data, std::vector<uint8_t>& activations, uint32_t amtOfThreads = 1, float eps = 0 /*maximum distance from data*/, bool andBrushes = false /*If true point has to be in all ranges*/){
        // converting the range brushes to properly be able to check activation
        struct MM{float min, max;};
        std::vector<std::map<int, std::vector<MM>>> axisBrushes(rangeBrushes.size());
        std::vector<bool> axisActive(data.size(), false);
        for(int i: irange(rangeBrushes)){
            auto& b = axisBrushes[i];
            for(const auto& range: rangeBrushes[i]){
                b[range.axis].push_back({range.min, range.max});
                axisActive[range.axis] = true;
            }
        }
        auto threadExec = [&](size_t start, size_t end){
            for(size_t i: irange(start, end)){
                bool a{rangeBrushes.empty()};
                // Range brushes ------------------------------------------------------
                for(auto& brush: axisBrushes){
                    bool in{true};
                    for(auto [axis, ranges]: brush){
                        T d = (*data[axis])[i];
                        bool inRange{false};
                        for(auto& r: ranges){
                            if(d + eps >= r.min && d - eps <= r.max){
                                inRange = true;
                                break;
                            }
                        }
                        in &= inRange;
                        if(!in)
                            break;
                    }
                    if(andBrushes && !in){
                        a = false;
                        break;
                    }
                    else if(!andBrushes && in){
                        a = true;
                        break;
                    }
                }
                if(!andBrushes && a){
                    activations[i / 8] |= uint8_t(a) << i & 7;       //early out when data point is already accepted
                    continue;
                }

                // Lasso brushes ------------------------------------------------------
                for(const auto& lasso: lassoBrushes){
                    int attr1 = lasso.attr1;
                    int attr2 = lasso.attr2;
                    bool inLasso = false;
                    ImVec2 d{static_cast<float>((*data[attr1])[i]), static_cast<float>((*data[attr2])[i])};
                    for(int j: irange(lasso.borderPoints)){
                        const ImVec2& a = lasso.borderPoints[j];
                        const ImVec2& b = lasso.borderPoints[(j + 1) % lasso.borderPoints.size()];
                        // calculate line intersection with horizontal line, code from https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
                        if( ((a.y > d.y) != (b.y > d.y)) &&
		        		    (d.x < (b.x - a.x) * (d.y - a.y) / (b.y - a.y) + a.x) )
		        		    inLasso = !inLasso;
                    }
                    a = a && inLasso;
                    if(!a)
                        break;
                }

                if(andBrushes)
                    activations[i / 8] &= uint8_t(a) << i & 7;
                else
                    activations[i / 8] |= uint8_t(a) << i & 7;
            }
        };
        if(amtOfThreads == 1){
            threadExec(0, data[0]->size());
        }
        else{
            std::vector<std::thread> threads(amtOfThreads);
            size_t curStart = 0;
            size_t size = data[0]->size();
            for(int i: irange(amtOfThreads)){
                size_t curEnd = size_t(i + 1) * size / amtOfThreads;
                threads[i] = std::thread(threadExec, curStart, curEnd);
                curStart = curEnd;
            }
            for(auto& t: threads)
                t.join();
        }
    }
}