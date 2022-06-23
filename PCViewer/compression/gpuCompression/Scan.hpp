#include "../cpuCompression/global.h"

#include <cstdlib>

#include "ScanPlan.hpp"
#include "GpuInstance.hpp"

namespace vkCompress {

// vulkan version doing the conversion from uint16 counts to uint offsets (inklusive)
template<bool isExclusive>
void scanArray(GpuInstance* pInstance, VkCommandBuffer commands, size_t start, size_t end, const ScanPlan* plan, uint level = 0)
{
    const uint scanElementsPerThread = 8;
    const uint scanCtaSize = 128;
    const uint blockSize = scanElementsPerThread * scanCtaSize;
    uint numBlocks = (uint(end - start) + blockSize - 1) / blockSize;

    uint sharedEltsPerBlock = scanCtaSize * 2;

    uint sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

    bool fullBlock = (end - start == numBlocks * scanElementsPerThread * scanCtaSize);

    // setup execution parameters
    uint dispatchX = numBlocks;

    struct PushConstants{
        uint numElements;
        uint dataRowPitch;
        uint blockSumRowPitch;
        uint srcUVec4;  // if 1, src4u should be used for loading
    }pc{};

    pc.numElements = end - start;
    pc.srcUVec4 = level != 0;

    if constexpr(isExclusive){
        vkCmdPushConstants(commands, pInstance->RunLength.inclusiveScanInfo.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->RunLength.inclusiveScanInfo.pipelineLayout, 0, 1, &plan->m_blockSets[level], {});
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->RunLength.inclusiveScanInfo.pipeline);
    }
    else{
        vkCmdPushConstants(commands, pInstance->RunLength.exclusiveScanInfo.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->RunLength.exclusiveScanInfo.pipelineLayout, 0, 1, &plan->m_blockSets[level], {});
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pInstance->RunLength.exclusiveScanInfo.pipeline);
    }
    vkCmdDispatch(commands, dispatchX, 1, 1);

    if (numBlocks > 1)
    {
        // After scanning all the sub-blocks, we are mostly done. But
        // now we need to take all of the last values of the
        // sub-blocks and scan those. This will give us a new value
        // that must be added to each block to get the final results.

        // recursive (CPU) call
        scanArray<true> // second-level scan always uses identity functor
            (pInstance, commands, start, numBlocks, plan, level + 1); 
        
        if (fullBlock) {
            vectorAddUniform4<TOut, Op, SCAN_ELTS_PER_THREAD, true>
                <<< grid, threads, 0, stream >>>(d_out,
                                                 (const TOut*)plan->m_blockSums[level],
                                                 (uint)numElements,
                                                 (uint)rowPitch*4,
                                                 blockSumRowPitch*4,
                                                 0, 0);
        } else {
            vectorAddUniform4<TOut, Op, SCAN_ELTS_PER_THREAD, false>
                <<< grid, threads, 0, stream >>>(d_out,
                                                 (const TOut*)plan->m_blockSums[level],
                                                 (uint)numElements,
                                                 (uint)rowPitch*4,
                                                 blockSumRowPitch*4,
                                                 0, 0);
        }
       
    }
}

/** @brief Perform recursive scan on arbitrary size arrays
  *
  * This is the CPU-side workhorse function of the scan engine.  This function
  * invokes the CUDA kernels which perform the scan on individual blocks. 
  *
  * Scans of large arrays must be split (possibly recursively) into a hierarchy of block scans,
  * where each block is scanned by a single CUDA thread block.  At each recursive level of the
  * scanArray first invokes a kernel to scan all blocks of that level, and if the level
  * has more than one block, it calls itself recursively.  On returning from each recursive level,
  * the total sum of each block from the level below is added to all elements of the corresponding
  * block in this level.  See "Parallel Prefix Sum (Scan) in CUDA" for more information (see
  * \ref references ).
  *
  * @param[out] d_out       The output array for the scan results
  * @param[in]  d_in        The input array to be scanned
  * @param[in]  numElements The number of elements in the array to scan
  * @param[in]  plan        A pointer to the plan structure for the reduction.
  * @param[in]  level       The current recursive level of the scan
  */
template <typename TIn, typename TOut, bool isExclusive, typename Func>
void scanArray(TOut* d_out, const TIn* d_in, size_t numElements, size_t numRows, size_t rowPitch, const ScanPlan* plan, cudaStream_t stream = 0, int level = 0)
{
    typedef OperatorAdd<TOut> Op;

    uint blockSize = SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE;
    uint numBlocks = (uint(numElements) + blockSize - 1) / blockSize;

    uint sharedEltsPerBlock = SCAN_CTA_SIZE * 2;

    uint sharedMemSize = sizeof(TOut) * sharedEltsPerBlock;

    // divide pitch by four since scan's load/store addresses are for vec4 elements
    uint blockSumRowPitch = 1;
    if (numRows > 1)
    {
        rowPitch         = rowPitch / 4;
        blockSumRowPitch = (uint)((numBlocks > 1) ? plan->m_rowPitches[level+1] / 4 : 0);
    }

    bool fullBlock = (numElements == numBlocks * SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE);

    // setup execution parameters
    dim3  grid(numBlocks, (uint)numRows, 1); 
    dim3  threads(SCAN_CTA_SIZE, 1, 1);

    // make sure there are no CUDA errors before we start
    cudaCheckMsg("scanArray before kernels");

    uint traitsCode = 0;
    if (numBlocks > 1) traitsCode |= 1;
    if (numRows > 1)   traitsCode |= 2;
    if (fullBlock)     traitsCode |= 4;

    switch (traitsCode)
    {
    case 0: // single block, single row, non-full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, false, false, false> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, nullptr, (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 1: // multiblock, single row, non-full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, false, true, false> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, (TOut*)plan->m_blockSums[level], (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 2: // single block, multirow, non-full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, true, false, false> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, nullptr, (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 3: // multiblock, multirow, non-full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, true, true, false> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, (TOut*)plan->m_blockSums[level], (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 4: // single block, single row, full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, false, false, true> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, nullptr, (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 5: // multiblock, single row, full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, false, true, true> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, (TOut*)plan->m_blockSums[level], (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 6: // single block, multirow, full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, true, false, true> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, nullptr, (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    case 7: // multiblock, multirow, full block
        scan4<TIn, TOut, ScanTraits<Op, Func, isExclusive, true, true, true> >
               <<< grid, threads, sharedMemSize, stream >>>
               (d_out, d_in, (TOut*)plan->m_blockSums[level], (uint)numElements, (uint)rowPitch, blockSumRowPitch);
        break;
    }

    cudaCheckMsg("prescan");

    if (numBlocks > 1)
    {
        // After scanning all the sub-blocks, we are mostly done. But
        // now we need to take all of the last values of the
        // sub-blocks and scan those. This will give us a new value
        // that must be added to each block to get the final results.

        // recursive (CPU) call
        scanArray<TOut, TOut, true, FunctorIdentity<TOut>> // second-level scan always uses identity functor
            ((TOut*)plan->m_blockSums[level], (const TOut*)plan->m_blockSums[level], numBlocks, numRows, numRows > 1 ? plan->m_rowPitches[level + 1] : 1, plan, stream, level + 1); 
        
        if (fullBlock) {
            vectorAddUniform4<TOut, Op, SCAN_ELTS_PER_THREAD, true>
                <<< grid, threads, 0, stream >>>(d_out,
                                                 (const TOut*)plan->m_blockSums[level],
                                                 (uint)numElements,
                                                 (uint)rowPitch*4,
                                                 blockSumRowPitch*4,
                                                 0, 0);
        } else {
            vectorAddUniform4<TOut, Op, SCAN_ELTS_PER_THREAD, false>
                <<< grid, threads, 0, stream >>>(d_out,
                                                 (const TOut*)plan->m_blockSums[level],
                                                 (uint)numElements,
                                                 (uint)rowPitch*4,
                                                 blockSumRowPitch*4,
                                                 0, 0);
        }
       
        cudaCheckMsg("vectorAddUniform");
    }
}

// convenience wrappers:

// single-row with given functor
template <typename TIn, typename TOut, bool isExclusive, typename Func>
void scanArray(TOut* d_out, const TIn* d_in, size_t numElements, const ScanPlan* plan, cudaStream_t stream = 0)
{
    assert(numElements <= plan->m_numElements);
    scanArray<TIn, TOut, isExclusive, Func>(d_out, d_in, numElements, 1, 1, plan, stream);
}

// single-row with identity functor
template <typename TIn, typename TOut, bool isExclusive>
void scanArray(TOut* d_out, const TIn* d_in, size_t numElements, const ScanPlan* plan, cudaStream_t stream = 0)
{
    scanArray<TIn, TOut, isExclusive, FunctorIdentity<TIn>>(d_out, d_in, numElements, plan, stream);
}

// multi-row with identity functor
template <typename TIn, typename TOut, bool isExclusive>
void scanArray(TOut* d_out, const TIn* d_in, size_t numElements, size_t numRows, size_t rowPitch, const ScanPlan* plan, cudaStream_t stream = 0)
{
    assert(numElements <= plan->m_numElements);
    assert(numRows <= plan->m_numRows);
    assert(numRows <= 1 || rowPitch <= plan->m_rowPitches[0]);
    scanArray<TIn, TOut, isExclusive, FunctorIdentity<TIn>>(d_out, d_in, numElements, numRows, rowPitch, plan, stream);
}

}