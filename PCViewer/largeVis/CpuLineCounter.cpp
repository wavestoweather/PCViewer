#include "CpuLineCounter.hpp"
#include <inttypes.h>
#include <atomic>
#include <vector>
#include <thread>
#include <random>
#include "../PCUtil.h"
#include "../range.hpp"

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

    std::vector<uint32_t> lineCounterPair(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, uint32_t amtOfThreads){
        std::vector<std::thread> threads(amtOfThreads);
        std::vector<std::vector<uint32_t>> lineCounts(amtOfThreads, std::vector<uint32_t>(aBins * bBins, 0));   // for each thread one vector is available which is initialized to 0
        
        auto threadExec = [&](uint32_t tId, uint32_t begin, uint32_t end){
            auto& localCounts = lineCounts[tId];
            for(auto cur = begin; cur != end; ++cur){
                int binA = aVals[cur] * (static_cast<int>(aBins) - 1) + .5f;
                int binB = bVals[cur] * (static_cast<int>(bBins) - 1) + .5f;
                //safety check
                binA %= aBins;
                binB %= bBins;
                ++localCounts[binA * bBins + binB];
            }
        };

        size_t size = aVals.size();
        PCUtil::Stopwatch stopwatch(std::cout, "CpuLineCounter counting time");
        for(uint32_t cur = 0; cur < amtOfThreads; ++cur){
            uint32_t begin = size_t(cur) * size / amtOfThreads;
            uint32_t end = size_t(cur + 1) * size / amtOfThreads;
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
        return lineCounts[0];
    }
}