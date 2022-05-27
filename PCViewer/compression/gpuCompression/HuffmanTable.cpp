#include "HuffmanTable.h"

namespace vkCompress{
using namespace cudaCompress;
HuffmanDecodeTable::HuffmanDecodeTable(const GpuInstance* pInstance)
{
    size_t storageSize = computeMaxGPUSize(context);
    m_pStorage = new byte[storageSize];

    cudaCompress::byte* pNext = m_pStorage;
    m_pCodewordFirstIndexPerLength = reinterpret_cast<int*>(pNext); pNext += MAX_CODEWORD_BITS * sizeof(int);
    m_pCodewordMinPerLength        = reinterpret_cast<int*>(pNext); pNext += MAX_CODEWORD_BITS * sizeof(int);
    m_pCodewordMaxPerLength        = reinterpret_cast<int*>(pNext); pNext += MAX_CODEWORD_BITS * sizeof(int);
    m_pSymbolTable                 = reinterpret_cast<byte*>(pNext);

    clear();

    // TODO: vulkan upload sync event
}

HuffmanDecodeTable::HuffmanDecodeTable(HuffmanDecodeTable&& other)
{
    // copy state
    memcpy(this, &other, sizeof(HuffmanDecodeTable));

    // clear other
    memset(&other, 0, sizeof(HuffmanDecodeTable));
}

HuffmanDecodeTable::~HuffmanDecodeTable()
{
    // TODO: free vulkan upload sync
    delete[] m_pStorage;
}

HuffmanDecodeTable& HuffmanDecodeTable::operator=(HuffmanDecodeTable&& other)
{
    if(this == &other)
        return *this;

    // release our own resources
    delete[] m_pStorage;

    // copy state from other
    memcpy(this, &other, sizeof(HuffmanDecodeTable));

    // clear other
    memset(&other, 0, sizeof(HuffmanDecodeTable));

    return *this;
}

void HuffmanDecodeTable::clear()
{
    m_symbolTableSize = 0;

    for(uint i = 0; i < MAX_CODEWORD_BITS; i++) {
        m_pCodewordFirstIndexPerLength[i] = -1;
    }
    for(uint i = 0; i < MAX_CODEWORD_BITS; i++) {
        m_pCodewordMinPerLength[i] = -1;
    }
    for(uint i = 0; i < MAX_CODEWORD_BITS; i++) {
        m_pCodewordMaxPerLength[i] = -1;
    }
}

void HuffmanDecodeTable::readFromBitStream(const GpuInstance* pInstance, BitStreamReadOnly& bitstream)
{
    clear();

    // read codeword count per length
    std::vector<uint> codewordCountPerLength;
    uint codewordCountPerLengthSize = 0;
    bitstream.readBits(codewordCountPerLengthSize, LOG2_MAX_CODEWORD_BITS);
    codewordCountPerLength.reserve(codewordCountPerLengthSize);
    for(uint i = 0; i < codewordCountPerLengthSize; i++) {
        uint codewordCount = 0;
        bitstream.readBits(codewordCount, LOG2_HUFFMAN_DISTINCT_SYMBOL_COUNT_MAX);
        codewordCountPerLength.push_back(codewordCount);
    }

    // read symbol table
    m_symbolTableSize = 0;
    bitstream.readBits(m_symbolTableSize, LOG2_HUFFMAN_DISTINCT_SYMBOL_COUNT_MAX);
    bool longSymbols = (LOG2_HUFFMAN_DISTINCT_SYMBOL_COUNT_MAX > 16);
    uint symbolBits = 0;
    bitstream.readBits(symbolBits, longSymbols ? LOG2_MAX_SYMBOL32_BITS : LOG2_MAX_SYMBOL16_BITS);
    // HACK: if symbolBits was 16, then 0 was written to the bitstream (4 least significant bits)
    if(!longSymbols && symbolBits == 0 && m_symbolTableSize > 1) symbolBits = 16;
    for(uint i = 0; i < m_symbolTableSize; i++) {
        uint symbol = 0;
        bitstream.readBits(symbol, symbolBits);
        if(longSymbols) {
            ((Symbol32*)m_pSymbolTable)[i] = Symbol32(symbol);
        } else {
            ((Symbol16*)m_pSymbolTable)[i] = Symbol16(symbol);
        }
    }

    build(codewordCountPerLength);
}

uint HuffmanDecodeTable::computeMaxGPUSize(const GpuInstance* pInstance)
{
    uint distinctSymbolCountMax = 1 << LOG2_HUFFMAN_DISTINCT_SYMBOL_COUNT_MAX;

    bool longSymbols = (LOG2_HUFFMAN_DISTINCT_SYMBOL_COUNT_MAX > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint size = 0;

    size += 3 * MAX_CODEWORD_BITS * sizeof(int);
    size += distinctSymbolCountMax * symbolSize;

    return size;
}

uint HuffmanDecodeTable::computeGPUSize(const GpuInstance* pInstance) const
{
    bool longSymbols = (LOG2_HUFFMAN_DISTINCT_SYMBOL_COUNT_MAX > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint size = 0;

    size += 3 * MAX_CODEWORD_BITS * sizeof(int);
    size += m_symbolTableSize * symbolSize;

    return size;
}

void HuffmanDecodeTable::copyToBuffer(const GpuInstance* pInstance, byte* pTable) const
{
    size_t size = computeGPUSize(context);
    memcpy(pTable, m_pStorage, size);
}

void HuffmanDecodeTable::uploadToGPU(const GpuInstance* pInstance, byte* dpTable) const
{
    size_t size = computeGPUSize(context);
    //cudaSafeCall(cudaMemcpy(dpTable, m_pStorage, size, cudaMemcpyHostToDevice));
    // TODO: upload to gpu
}

void HuffmanDecodeTable::uploadToGPUAsync(const VkUtil::Context* context, byte* dpTable) const
{
    size_t size = computeGPUSize(context);
    //cudaSafeCall(cudaMemcpyAsync(dpTable, m_pStorage, size, cudaMemcpyHostToDevice, pInstance->m_stream));
    //cudaSafeCall(cudaEventRecord(m_uploadSyncEvent, pInstance->m_stream));
    // TDOO: upload async
}

void HuffmanDecodeTable::syncOnLastAsyncUpload() const
{
    //cudaSafeCall(cudaEventSynchronize(m_uploadSyncEvent));
    // TODO: check if upload went through
}

void HuffmanDecodeTable::build(const std::vector<uint>& codewordCountPerLength)
{
    if(m_symbolTableSize == 0)
        return;

    // count total number of codewords
    uint codewordCount = 0;
    for(uint i = 0; i < codewordCountPerLength.size(); i++) {
        codewordCount += codewordCountPerLength[i];
    }
    if(codewordCountPerLength.empty()) {
        // this can happen when all symbols are the same -> only a single "codeword" with length 0
        codewordCount++;
    }

    assert(codewordCount == m_symbolTableSize);

    // find codeword lengths
    std::vector<uint> codewordLengths;
    codewordLengths.reserve(codewordCount);
    for(uint i = 0; i < codewordCountPerLength.size(); i++) {
        codewordLengths.insert(codewordLengths.cend(), codewordCountPerLength[i], i + 1);
    }

    // find codewords
    std::vector<int> codewords;
    codewords.reserve(codewordCount);

    codewords.push_back(0);
    for(uint index = 1; index < codewordCount; index++) {
        // new codeword = increment previous codeword
        int codeword = codewords[index-1] + 1;
        // append zero bits as required to reach correct length
        uint lengthDiff = codewordLengths[index] - codewordLengths[index-1];
        codeword <<= lengthDiff;

        codewords.push_back(codeword);
    }

    // build indices (by codeword length) into table
    uint codewordLengthMax = uint(codewordCountPerLength.size());
    assert(codewordLengthMax <= MAX_CODEWORD_BITS);
    // loop over codeword lengths (actually (length-1))
    for(uint codewordLength = 0, entry = 0; codewordLength < codewordLengthMax; codewordLength++) {
        if(codewordCountPerLength[codewordLength] > 0) {
            // current entry is first codeword of this length
            m_pCodewordFirstIndexPerLength[codewordLength] = entry;
            // store value of first codeword of this length
            m_pCodewordMinPerLength[codewordLength] = codewords[entry];
            // move to last codeword of this length
            entry += codewordCountPerLength[codewordLength] - 1;
            // store value of last codeword of this length
            m_pCodewordMaxPerLength[codewordLength] = codewords[entry];
            // move to first codeword of next length
            entry++;
        } else {
            m_pCodewordFirstIndexPerLength[codewordLength] = -1;
            m_pCodewordMinPerLength[codewordLength] = -1;
            m_pCodewordMaxPerLength[codewordLength] = -1;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

size_t HuffmanEncodeTable::getRequiredMemory(const GpuInstance* pInstance)
{
    uint tableCountMax = pInstance->m_streamCountMax;
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    size_t size = 0;

    // dpHistograms
    for(uint i = 0; i < tableCountMax; i++) {
        size += getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);
    }

    // dpReduceOut
    size += getAlignedSize(tableCountMax * sizeof(Symbol32), 128);

    return size;
}

void HuffmanEncodeTable::init(VkUtil::Context* context)
{
    uint tableCountMax = pInstance->m_streamCountMax;
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint distinctSymbolCountMaxAligned = (uint)getAlignedSize(distinctSymbolCountMax, 128 / sizeof(uint));

    cudaSafeCall(cudaMallocHost(&pInstance->HuffmanTable.pReadback, tableCountMax * distinctSymbolCountMaxAligned * sizeof(uint)));
}

void HuffmanEncodeTable::shutdown(Instance* pInstance)
{
    cudaSafeCall(cudaFreeHost(pInstance->HuffmanTable.pReadback));
    pInstance->HuffmanTable.pReadback = nullptr;
}


HuffmanEncodeTable::HuffmanEncodeTable(const VkUtil::Context* context)
    : m_symbolMax(0), m_codewordTableSize(0)
{
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;
    uint codewordTableSizeMax = distinctSymbolCountMax;
    //cudaSafeCall(cudaMallocHost(&m_pCodewords,       codewordTableSizeMax * sizeof(uint), cudaHostAllocWriteCombined));
    //cudaSafeCall(cudaMallocHost(&m_pCodewordLengths, codewordTableSizeMax * sizeof(uint), cudaHostAllocWriteCombined));
    m_pCodewords = new uint[codewordTableSizeMax];
    m_pCodewordLengths = new uint[codewordTableSizeMax];
}

HuffmanEncodeTable::HuffmanEncodeTable(HuffmanEncodeTable&& other)
{
    m_symbolMax = other.m_symbolMax;
    m_symbols.swap(other.m_symbols);
    m_codewordCountPerLength.swap(other.m_codewordCountPerLength);

    m_pCodewords = other.m_pCodewords;
    other.m_pCodewords = nullptr;
    m_pCodewordLengths = other.m_pCodewordLengths;
    other.m_pCodewordLengths = nullptr;
    m_codewordTableSize = other.m_codewordTableSize;
    other.m_codewordTableSize = 0;
}

HuffmanEncodeTable::~HuffmanEncodeTable()
{
    clear();

    //cudaSafeCall(cudaFreeHost(m_pCodewords));
    //cudaSafeCall(cudaFreeHost(m_pCodewordLengths));
    delete[] m_pCodewords;
    delete[] m_pCodewordLengths;
}

}