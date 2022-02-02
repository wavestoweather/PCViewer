#include "DWTCpu.h"
#include <algorithm>

namespace cudaCompress::util
{
    #if CUDACOMPRESS_DWT_FLOAT_FILTER == 0
    #define FILTER_LENGTH 9
    #define FILTER_OFFSET 4
    #elif CUDACOMPRESS_DWT_FLOAT_FILTER == 1
    #define FILTER_LENGTH 5
    #define FILTER_OFFSET 2
    #elif CUDACOMPRESS_DWT_FLOAT_FILTER == 2
    #define FILTER_LENGTH 5
    #define FILTER_OFFSET 2
    #elif CUDACOMPRESS_DWT_FLOAT_FILTER == 3
    #define FILTER_LENGTH 3
    #define FILTER_OFFSET 1
    #endif

    #define FILTER_OFFSET_RIGHT (FILTER_LENGTH - FILTER_OFFSET - 1)

    #define COEFFICIENT_COUNT (2 * FILTER_LENGTH)

    #define FILTER_OVERHEAD_LEFT_INV ((FILTER_OFFSET+1)/2*2)
    #define FILTER_OVERHEAD_RIGHT_INV ((FILTER_OFFSET_RIGHT+1)/2*2)
    #define FILTER_OVERHEAD_INV (FILTER_OVERHEAD_LEFT_INV + FILTER_OVERHEAD_RIGHT_INV)

    // The forward filters are normalized so that the (nominal) analysis gain is 1 for both low-pass and high-pass.
    // (according to final draft March 2000, JPEG2000 uses a gain of 2 in the high-pass filter and adjusts the quantization steps accordingly)
    // (however, "JPEG 2000 image compression fundamentals, standards and practice" lists the numbers below)
    #if CUDACOMPRESS_DWT_FLOAT_FILTER == 0
    static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
        0.0267488f,-0.0168641f,-0.0782233f, 0.2668641f, 0.6029490f, 0.2668641f,-0.0782233f,-0.0168641f, 0.0267488f, // low-pass
        0.0f,       0.0456359f,-0.0287718f,-0.2956359f, 0.5575435f,-0.2956359f,-0.0287718f, 0.0456359f, 0.0f        // high-pass
    };
    static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
        0.0f,       0.0337282f,-0.0575435f,-0.5337281f, 1.1150871f,-0.5337281f,-0.0575435f, 0.0337282f, 0.0f,       // even (interleaved lp and hp)
        0.0534975f,-0.0912718f,-0.1564465f, 0.5912718f, 1.2058980f, 0.5912718f,-0.1564465f,-0.0912718f, 0.0534975f  // odd  (interleaved hp and lp)
    };
    #elif CUDACOMPRESS_DWT_FLOAT_FILTER == 1
    static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
        -0.125f, 0.25f, 0.75f, 0.25f,-0.125f,
         0.0f,  -0.25f, 0.5f, -0.25f, 0.0f
    };
    static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
         0.0f,  -0.5f,  1.0f, -0.5f,  0.0f,
        -0.25f,  0.5f,  1.5f,  0.5f, -0.25f
    };
    #elif CUDACOMPRESS_DWT_FLOAT_FILTER == 2
    static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
         0.0f,       0.34150635f, 0.59150635f,  0.15849365f, -0.091506f,
        -0.091506f, -0.15849365f, 0.59150635f, -0.34150635f,  0.0f
    };
    static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
        -0.183012f, -0.6830127f, 1.1830127f, -0.3169873f,  0.0f,
         0.0f,       0.3169873f, 1.1830127f,  0.6830127f, -0.183012f
    };
    #elif CUDACOMPRESS_DWT_FLOAT_FILTER == 3
    static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
         0.0f, 0.5f, 0.5f,
        -0.5f, 0.5f, 0.0f
    };
    static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
         0.0f, 1.0f,-1.0f,
         1.0f, 1.0f, 0.0f
    };
    #endif

    void dwtFloatForwardCPU(float* dst, float* src, int size, int dstRowPitch, int srcRowPitch) 
    {
        int highPassOffset = size / 2;

        for(int i = 0; i < size; ++i){
            const int highpass = i & 1;
            const int filterBaseIndex = highpass * FILTER_LENGTH;
            float sum = 0;
            for(int f = 0; f < FILTER_LENGTH; ++f){
                int index = i + f - FILTER_LENGTH / 2;
                index = std::abs(index);                                                //standard mirror left
                index = index >= size ? (size-1) - std::abs(size-1 - index): index;     //standard mirror right
                sum += g_ForwardFilterCoefficients[filterBaseIndex + f] * src[index + srcRowPitch];
            }
            dst[highpass * highPassOffset + i / 2 + dstRowPitch] = sum;
        }
    }

    void dwtFloatInverseCPU(float* dst, float* src, int size, int dstRowPitch, int srcRowPitch){
        for(int i = 0; i < size; ++i){
            const int highpass = i & 1;
            const int filterBaseIndex = highpass * FILTER_LENGTH;
            float sum = 0;
            for(int f = 0; f < FILTER_LENGTH; ++f){
                int index = i + f - FILTER_LENGTH / 2;
                if(index < 0)// && !(index & 1))
                    index = std::abs(index);                                                 //standard mirror left
                if(index >= size)
                    index = (size-1) - std::abs(size-1 - index);                             //standard mirror right
                index = (index >> 1) + ((index & 1) * (size >> 1)); //interleaving the data with first low pass coeffs and then high pass coeffs
                sum += g_InverseFilterCoefficients[filterBaseIndex + f] * src[index + srcRowPitch];
            }
            dst[i + dstRowPitch] = sum;
        }
    }

    void quantizeToSymbols(ushort* dpSymbols, const float* dpData, uint size, float quantizationStep, uint rowPitchSrc, EQuantizeType quantType){
        float quantInv = 1.0f / quantizationStep;
        switch(quantType){
            case QUANTIZE_DEADZONE:
                for(int i = 0; i < size; ++i){
                    float val = dpData[i];
                    int quantized = val * quantInv;
                    uint32_t symbolized = 2 * std::abs(quantized) + (quantized >> 31);
                    dpSymbols[i] = ushort(symbolized);
                }
                break;
            case QUANTIZE_UNIFORM:
                for(int i = 0; i < size; ++i){
                    float val = dpData[i];
                    int quantized = val * quantInv + .5f * (val < 0 ? -1.f : val > 0 ? 1.f : 0);
                    uint32_t symbolized = 2 * std::abs(quantized) + (quantized >> 31);
                    dpSymbols[i] = ushort(symbolized);
                }
                break;
        }
    }

    void unquantizeFromSymbols(float* dpData, const ushort* dpSymbols, uint size, float quantizationStep, uint rowPitchDst, EQuantizeType quantType){
        switch(quantType){
            case QUANTIZE_DEADZONE:
                for(int i = 0; i < size; ++i){
                    auto symbol = dpSymbols[i];
                    int neg = symbol & 1;
                    int unsymbolized = (1 - 2 * neg) * ((symbol + neg) / 2);
                    dpData[i] = (float(unsymbolized) + .5f * (unsymbolized < 0 ? -1.f : unsymbolized > 0 ? 1.f : .0f)) * quantizationStep;
                }
                break;
            case QUANTIZE_UNIFORM:
                for(int i = 0; i < size; ++i){
                    auto symbol = dpSymbols[i];
                    int neg = symbol & 1;
                    int unsymbolized = (1 - 2 * neg) * ((symbol + neg) / 2);
                    dpData[i] = float(unsymbolized) * quantizationStep;
                }
                break;
        }
    }
}