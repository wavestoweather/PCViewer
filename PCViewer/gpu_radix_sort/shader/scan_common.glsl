#include "radix_common.glsl"

shared uint gs_FFX_PARALLELSORT_LDS[FFX_PARALLELSORT_ELEMENTS_PER_THREAD][FFX_PARALLELSORT_THREADGROUP_SIZE];
void FFX_ParallelSort_ScanPrefix(uint numValuesToScan, uint localID, uint groupID, uint BinOffset, uint BaseIndex, bool AddPartialSums,
                                    uint_vec ScanSrc, uint_vec ScanDst, uint_vec ScanScratch)
{
    uint i;
    // Perform coalesced loads into LDS
    for (i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        uint DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;

        uint col = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) / FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        uint row = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) % FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        gs_FFX_PARALLELSORT_LDS[row][col] = (DataIndex < numValuesToScan) ? ScanSrc.d[BinOffset + DataIndex] : 0;
    }

    // Wait for everyone to catch up
    barrier();

    uint threadgroupSum = 0;
    // Calculate the local scan-prefix for current thread
    for (i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        uint tmp = gs_FFX_PARALLELSORT_LDS[i][localID];
        gs_FFX_PARALLELSORT_LDS[i][localID] = threadgroupSum;
        threadgroupSum += tmp;
    }

    // Scan prefix partial sums
    threadgroupSum = FFX_ParallelSort_BlockScanPrefix(threadgroupSum, localID);

    // Add reduced partial sums if requested
    uint partialSum = 0;
    if (AddPartialSums)
    {
        // Partial sum additions are a little special as they are tailored to the optimal number of 
        // thread groups we ran in the beginning, so need to take that into account
        partialSum = ScanScratch.d[groupID];
    }

    // Add the block scanned-prefixes back in
    for (i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        gs_FFX_PARALLELSORT_LDS[i][localID] += threadgroupSum;

    // Wait for everyone to catch up
    barrier();

    // Perform coalesced writes to scan dst
    for (i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        uint DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;

        uint col = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) / FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        uint row = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) % FFX_PARALLELSORT_ELEMENTS_PER_THREAD;

        if (DataIndex < numValuesToScan)
            ScanDst.d[BinOffset + DataIndex] = gs_FFX_PARALLELSORT_LDS[row][col] + partialSum;
    }
}