#pragma once
#include <inttypes.h>

namespace cudaCompress::util{
    // 0 : CDF 9/7
    // 1 : CDF 5/3
    // 2 : Daubechies 4 (with broken boundaries, because we use mirrored extension)
    // 3 : Haar
    #define CUDACOMPRESS_DWT_FLOAT_FILTER 0

    enum EQuantizeType
    {
        QUANTIZE_DEADZONE = 0, // midtread quantizer with twice larger zero bin
        QUANTIZE_UNIFORM,      // standard uniform midtread quantizer
        QUANTIZE_COUNT
    };

    using ushort = uint16_t;
    using uint = uint32_t;

    /*
        dstRowPitch is the offset from the start of the dst array from which 'size' elements will be transformed.
        same goes for srcRowPitch for the src array
    */
    void dwtFloatForwardCPU(float* dst, float* src, int size, int dstRowPitch = 0, int srcRowPitch = 0);
    void dwtFloatInverseCPU(float* dst, float* src, int size, int dstRowPitch = 0, int srcRowPitch = 0);

    void quantizeToSymbols(ushort* dpSymbols, const float* dpData, uint size, float quantizationStep, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
    void unquantizeFromSymbols(float* dpData, const ushort* dpSymbols, uint size, float quantizationStep, uint rowPitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
}