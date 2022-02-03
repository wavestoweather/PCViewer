#include "HirarchyCreation.hpp"

#include "LeaderNode.hpp"
#include "../rTree/RTreeDynamic.h"
#include "cpuCompression/EncodeCPU.h"
#include "cpuCompression/DWTCpu.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>
#include <queue>
#include <atomic>
#include <future>
#include <fstream>
#include <memory>

namespace compression
{
    void createHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads, float quantizationStep) 
    {
        createTempHirarchy(outputFolder, loader, lvl0eps, levels, lvlMultiplier, maxMemoryMB, amtOfThreads);
        compressTempHirarchy(outputFolder, amtOfThreads, quantizationStep);
    }

    void createTempHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads) 
    {
        try{
            std::filesystem::current_path(outputFolder);
            //creating the temp folder for the temporary non compressed files
            std::filesystem::create_directory("temp/");
            std::string tempPath = std::string(outputFolder) + "/temp";

            std::vector<float> dataPoint;
            bool hasData = loader->getNextNormalized(dataPoint);
            if(!hasData){
                std::cout << "compression::createHirarchy(...) given loader has either no elements or was already loaded. Reset or insert loader with data elements" << std::endl;
                return;
            }

            // converting lvl multiplier to epsilon multiplier
            double epsMult = pow(1.0/lvlMultiplier, 1.0/dataPoint.size());
            std::unique_ptr<HierarchyCreateNode> root = std::make_unique<LeaderNode>(dataPoint, lvl0eps, epsMult, 0, levels);   //constructor automatically inserts the first data point
            std::shared_mutex cacheMutex;                            //mutex for the root node to control insert/cache access

            const int checkInterval = 1000;
            std::atomic<int> sizeCheck = checkInterval;
            auto threadFunc = [&](int threadId){
                std::vector<float> threadData;
                while(loader->getNextNormalized(threadData)){
                    //insert into the hirarchy
                    std::shared_lock<std::shared_mutex> insertLock(cacheMutex);
                    root->addDataPoint(threadData);
                    insertLock.unlock();

                    //should add caching strategies to avoid memory overflow and inbetween writeouts
                    if(--sizeCheck < 0 && threadId == 0){
                        std::unique_lock<std::shared_mutex> lock(cacheMutex); // locking the root node unique to do caching
                        sizeCheck = checkInterval;
                        size_t structureSize = root->getByteSize();
                        if(structureSize > maxMemoryMB * 1024 * 1024){
                            int dummy;
                            HierarchyCreateNode* cache = root->getCacheNode(dummy);
                            std::vector<float> half(.5f, threadData.size());
                            root->cacheNode(tempPath, "", half.data(), .5f, cache);
                        }
                    }
                }
            };
            {
                std::cout << "Creating all threds" << std::endl;
                std::vector<std::future<void>> threads(amtOfThreads);
                for(int i = 0; i < amtOfThreads; ++i){
                    threads[i] = std::async(threadFunc, i);
                }
                std::cout << "Threads up and running" << std::endl;
            }   // all futures are automatically joined at the end of the section
            std::cout << "Threads done with hierarchy creation" << std::endl;
            
            //final writeout to disk
            bool hellYeah = true;
            std::vector<float> half(dataPoint.size(), .5f);
            root->cacheNode(tempPath, "", half.data(), .5f, root.get());
            //info file containing 
            std::ofstream file(std::string(outputFolder) + "/attr.info", std::ios_base::binary);
            std::vector<Attribute> attributes;
            size_t tmp;
            loader->dataAnalysis(tmp, attributes);
            for(auto& a: attributes){
                file << a.name << " " << a.min << " " << a.max << "\n"; 
            }
        }
        catch(std::filesystem::filesystem_error err){
            std::cout << "Error trying to open output folder " << err.path1() << " with code: " << err.code() << std::endl;
        }
    }
    
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep) 
    {
        std::string tempPath = std::string(outputFolder) + "/temp";
        std::vector<std::string> cacheFiles;
        // getting all cache files
        for(const auto& entry: std::filesystem::directory_iterator(tempPath)){
            if(entry.is_regular_file() && entry.path().string().find('.') == std::string::npos){    // is cache file
                cacheFiles.push_back(entry.path().string());
            }
        }

        // compressing the cache files
        auto compressThread = [&](const std::vector<std::string>& files){
            for(auto& f: files){
                std::ifstream fs(f, std::ios_base::binary);
                std::vector<float> data;
                int rowLength;
                while(!fs.eof() && fs.good()){
                    int dataSize;
                    fs >> rowLength >> dataSize;
                    fs.get();   //newline char
                    //reading the data
                    std::vector<float> d(dataSize);
                    fs.read(reinterpret_cast<char*>(d.data()), dataSize * sizeof(d[0]));
                    data.insert(data.end(), d.begin(), d.end());
                    fs.get();   //newline char
                }
                //converting from row major to column major
                std::vector<float> col(data.size());
                uint32_t colInd = 0;
                for(uint32_t curInd = 0; colInd < data.size(); curInd += rowLength){
                    if(curInd >= data.size()) ++curInd %= data.size();
                    col[colInd++] = data[curInd];
                }
                //compressing the data with 2 dwts, followed by run-length and huffman encoding of quantized symbols
                //padding to size % 4 size
                uint originalLength = col.size();
                uint paddedLength = ((4 - (col.size() & 0b11)) & 0b11) + col.size();
                col.resize(paddedLength); data.resize(paddedLength);
                cudaCompress::util::dwtFloatForwardCPU(data.data(), col.data(), data.size(), 0, 0);
                std::copy(data.begin(), data.begin() + paddedLength / 2, col.begin());
                cudaCompress::util::dwtFloatForwardCPU(col.data(), data.data(), data.size() / 2, data.size() / 2, data.size() / 2);
                std::vector<cudaCompress::Symbol16> symbols(col.size());
                cudaCompress::util::quantizeToSymbols(symbols.data(), col.data(), col.size(), quantizationStep);
                cudaCompress::BitStream bitStream;
	            cudaCompress::BitStream* arr[]{&bitStream};
                std::vector<cudaCompress::Symbol16>* sArr[]{&symbols};
                cudaCompress::encodeRLHuffCPU(arr, sArr, 1, symbols.size());

                std::string outName = f.substr(f.find_last_of("/\\") + 1);  //should there be no /, then npos + 1  = 0, so the whole string is taken
                outName = std::string(outputFolder) + "/" + outName;
                std::ofstream out(outName);
                out << bitStream.getRawSizeBytes() << " " << symbols.size() << " " << originalLength << " " << quantizationStep << "\n";
                int s = bitStream.getRawSizeBytes();
                out.write(reinterpret_cast<char*>(bitStream.getRaw()), bitStream.getRawSizeBytes());
            }
        };

        {
            std::vector<std::future<void>> futures(amtOfThreads);
            auto curStart = cacheFiles.begin();
            for(int i = 0; i < amtOfThreads; ++i){
                auto curEnd = cacheFiles.begin() + (i + 1) * cacheFiles.size() / amtOfThreads;
                std::vector<std::string> subSet(curStart, curEnd);
                curStart = curEnd;
                futures[i] = std::async(compressThread, subSet);
            }
        }
    }
    
    void loadAndDecompress(const std::string_view& file, Data& data) 
    {
	    std::ifstream in(file.data());
	    uint byteSize, symbolsSize, dataSize;
	    float quantizationStep;
	    in >> byteSize >> symbolsSize >> dataSize >> quantizationStep;
	    in.get();	//skipping newline
	    std::vector<uint> bytes(byteSize / 4);
	    in.read(reinterpret_cast<char*>(bytes.data()), byteSize);
	    cudaCompress::BitStreamReadOnly bs(bytes.data(), byteSize * 8);
	    cudaCompress::BitStreamReadOnly* dec[]{&bs};
	    std::vector<cudaCompress::Symbol16> nS(symbolsSize);
	    std::vector<cudaCompress::Symbol16>* ss[]{&nS};
	    cudaCompress::decodeRLHuffCPU(dec, ss, symbolsSize, 1, symbolsSize);
	    std::vector<float> result(symbolsSize), result2(symbolsSize);
	    cudaCompress::util::unquantizeFromSymbols(result.data(), nS.data(), nS.size(), quantizationStep);
	    result2 = result;
	    cudaCompress::util::dwtFloatInverseCPU(result2.data(), result.data(), result.size() / 2, result.size() / 2, result.size() / 2);
	    cudaCompress::util::dwtFloatInverseCPU(result.data(), result2.data(), result.size());

        // getting attribute infos to get the columns amount
        std::string base(file.substr(0, file.find_last_of("/\\")));
        std::ifstream info(base + "/attr.info");
        int colCount = 0;
        while(!info.eof() && info.good()){
            std::string name;
            float min, max;
            info >> name >> min >> max;
            info.get(); //skip newline
            ++colCount;
        }
        data.columns.resize(colCount);
        data.columnDimensions.resize(colCount);
        uint32_t colSize = dataSize / colCount;
        data.dimensionSizes = {colSize};
        for(int i = 0; i < colCount; ++i){
            data.columns[i].resize(colSize);
            std::copy_n(result.begin() + i * colSize, colSize, data.columns[i].begin());
            data.columnDimensions[i] = {0};     //dependant only on the first dimension, which is the linear index dimension
        }   
    }
}