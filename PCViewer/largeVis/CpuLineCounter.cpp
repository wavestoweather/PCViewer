#include "CpuLineCounter.hpp"
#include <inttypes.h>
#include <atomic>
#include <vector>
#include <thread>
#include <random>
#include "../PCUtil.h"
#include "../range.hpp"

constexpr bool CheckCounts = false;

namespace compression
{
    void testCounting() 
    {
        std::cout << "Starting testCounting() setup ..." << std::endl;
        const uint32_t size = 1 << 30;
        const uint32_t aBins = 1 << 7, bBins = 1 << 7;
        const uint32_t amtOfThreads = 12;
        std::vector<std::atomic<uint32_t>> lineCounts(aBins * bBins);

        std::vector<uint16_t> aVals(size), bVals(size);
        //filling with random numbers
        std::srand(std::time(nullptr));
        for(auto& e: aVals) e = std::rand() & std::numeric_limits<uint16_t>::max();
        for(auto& e: bVals) e = std::rand() & std::numeric_limits<uint16_t>::max();

        std::vector<std::thread> threads(amtOfThreads);
        
        auto threadExec = [&](uint32_t begin, uint32_t end){
            for(auto cur = begin; cur != end; ++cur){
                int binA = float(aVals[cur]) / std::numeric_limits<uint16_t>::max() * aBins;
                int binB = float(bVals[cur]) / std::numeric_limits<uint16_t>::max() * bBins;
                binA %= aBins;
                binB %= bBins;
                ++lineCounts[binA * bBins + binB];
            }
        };

        std::cout << "Setup done, starting to count with " << amtOfThreads << " threads for " << size << " datapoints ..." << std::endl;
        PCUtil::Stopwatch stopwatch(std::cout, "CpuLineCounter counting time");
        for(uint32_t cur = 0; cur < amtOfThreads; ++cur){
            uint32_t begin = size_t(cur) * size / amtOfThreads;
            uint32_t end = size_t(cur + 1) * size / amtOfThreads;
            threads[cur] = std::thread(threadExec, begin, end);
        }
        // wait for all threads
        for(auto& t: threads)
            t.join();
        //uint32_t count{};
        //for(const auto& a: lineCounts)
        //    count += a;
        //bool hell = false;
    }

    std::vector<uint32_t> lineCounterPair(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, const std::vector<uint8_t>& activation, uint32_t amtOfThreads){
        std::vector<std::thread> threads(amtOfThreads);
        std::vector<std::vector<uint32_t>> lineCounts(amtOfThreads, std::vector<uint32_t>(aBins * bBins, 0));   // for each thread one vector is available which is initialized to 0
        
        auto threadExec = [&](uint32_t tId, size_t begin, size_t end){
            auto& localCounts = lineCounts[tId];
            for(auto cur = begin; cur != end; ++cur){
                size_t p = cur / 8;
                uint8_t bit = 1 << (cur & 7);
                if((activation[p] & bit) == 0)
                    continue;       // skip non active indices
                int binA = float(aVals[cur]) * (static_cast<int>(aBins) - 1) + .5f;
                int binB = float(bVals[cur]) * (static_cast<int>(bBins) - 1) + .5f;
                //safety check
                binA %= aBins;
                binB %= bBins;
                ++localCounts[binA * bBins + binB];
            }
        };

        size_t size = aVals.size();
        PCUtil::Stopwatch stopwatch(std::cout, "CpuLineCounter counting time");
        for(uint32_t cur = 0; cur < amtOfThreads; ++cur){
            size_t begin = size_t(cur) * size / amtOfThreads;
            size_t end = size_t(cur + 1) * size / amtOfThreads;
            threads[cur] = std::thread(threadExec, cur, begin, end);
        }
        // wait for all threads
        for(auto& t: threads)
            t.join();

        // summing up everything in the first vector
        for(uint32_t i: irange(1, lineCounts.size())){
            for(uint32_t e: irange(lineCounts[0])){
                lineCounts[0][e] += lineCounts[i][e];
            }
        }
        
        if constexpr(CheckCounts){
            size_t sum{};
            for(auto i: lineCounts[0])
                sum += i;
            std::cout << "Total count " << sum << std::endl;
        }
        return lineCounts[0];
    }

    std::vector<uint32_t> lineCounterPairSingleField(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, const std::vector<uint8_t>& activation, uint32_t amtOfThreads){
        std::vector<std::thread> threads(amtOfThreads);
        std::vector<std::atomic<uint32_t>> lineCounts(aBins * bBins);
        
        auto threadExec = [&](uint32_t tId, size_t begin, size_t end){
            auto& localCounts = lineCounts;
            for(auto cur = begin; cur != end; ++cur){
                size_t p = cur / 8;
                uint8_t bit = 1 << (cur & 7);
                if((activation[p] & bit) == 0)
                    continue;       // skip non active indices
                int binA = float(aVals[cur]) * (static_cast<int>(aBins) - 1) + .5f;
                int binB = float(bVals[cur]) * (static_cast<int>(bBins) - 1) + .5f;
                //safety check
                binA %= aBins;
                binB %= bBins;
                ++localCounts[binA * bBins + binB];
            }
        };

        size_t size = aVals.size();
        PCUtil::Stopwatch stopwatch(std::cout, "CpuLineCounter counting time");
        for(uint32_t cur = 0; cur < amtOfThreads; ++cur){
            size_t begin = size_t(cur) * size / amtOfThreads;
            size_t end = size_t(cur + 1) * size / amtOfThreads;
            threads[cur] = std::thread(threadExec, cur, begin, end);
        }
        // wait for all threads
        for(auto& t: threads)
            t.join();

        std::vector<uint32_t> ret(lineCounts.begin(), lineCounts.end());
        return ret;
    }

    std::vector<half> lineMinPair(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, const std::vector<uint8_t>& activation, const std::vector<half>& distances, uint32_t amtOfThreads){
        std::vector<std::thread> threads(amtOfThreads);
        std::vector<std::vector<half>> minDistances(amtOfThreads, std::vector<half>(aBins * bBins, .0f));   // for each thread one vector is available which is initialized to 0
        
        auto threadExec = [&](uint32_t tId, size_t begin, size_t end){
            auto& localMins = minDistances[tId];
            for(auto cur = begin; cur != end; ++cur){
                size_t p = cur / 8;
                uint8_t bit = 1 << (cur & 7);
                if((activation[p] & bit) == 0)
                    continue;       // skip non active indices
                int binA = float(aVals[cur]) * (static_cast<int>(aBins) - 1) + .5f;
                int binB = float(bVals[cur]) * (static_cast<int>(bBins) - 1) + .5f;
                half dist = distances[cur];
                //safety check
                binA %= aBins;
                binB %= bBins;
                size_t index = binA * bBins + binB;
                if(dist < localMins[index])
                    localMins[index] = dist;
            }
        };

        size_t size = aVals.size();
        PCUtil::Stopwatch stopwatch(std::cout, "CpuLineCounter counting time");
        for(uint32_t cur = 0; cur < amtOfThreads; ++cur){
            size_t begin = size_t(cur) * size / amtOfThreads;
            size_t end = size_t(cur + 1) * size / amtOfThreads;
            threads[cur] = std::thread(threadExec, cur, begin, end);
        }
        // wait for all threads
        for(auto& t: threads)
            t.join();

        // summing up everything in the first vector
        for(uint32_t i: irange(1, minDistances.size())){
            for(uint32_t e: irange(minDistances[0])){
                if(minDistances[i][e] < minDistances[0][e])
                minDistances[0][e] = minDistances[i][e];
            }
        }
        return minDistances[0];
    }
}