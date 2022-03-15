#include "HirarchyCreation.hpp"

#include "LeaderNode.hpp"
#include "HashNode.hpp"
#include "VectorLeaderNode.hpp"
#include "HashVectorLeaderNode.hpp"
#include "HashLeaderNode.hpp"
#include "../rTree/RTreeDynamic.h"
#include "cpuCompression/EncodeCPU.h"
#include "cpuCompression/DWTCpu.h"
#include "BundlingCacheManager.hpp"
#include "../PCUtil.h"
#include "../robin_hood_map/robin_hood.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>
#include <queue>
#include <atomic>
#include <future>
#include <fstream>
#include <memory>
#include <numeric>


//private functions ----------------------------------------------------------------------------
struct CacheInfo{std::string file; uint64_t offset, size;};
struct MutexPair{std::mutex info, data;};       //struct for holding two mutexes for the level files
struct CachingElementsInfo{uint32_t levelCount; robin_hood::unordered_map<std::string, std::vector<CacheInfo>> cacheElements;};

static void row2Col(const std::vector<float>& rowMajor, uint32_t rowLength, std::vector<float>& col){
    col.resize(rowMajor.size());
    uint32_t colInd = 0;
    for(uint32_t curInd = 0; colInd < rowMajor.size(); curInd += rowLength){
        if(curInd >= rowMajor.size()) ++curInd %= rowMajor.size();
        col[colInd++] = rowMajor[curInd];
    }
}

static void compressVector(std::vector<float>& src, float quantizationStep, /*out*/ cudaCompress::BitStream& bitStream, uint32_t& symbolsSize){
    //compressing the data with 2 dwts, followed by run-length and huffman encoding of quantized symbols
    //padding to size % 4 size
    uint32_t originalLength = src.size();
    uint32_t paddedLength = ((4 - (src.size() & 0b11)) & 0b11) + src.size();
    src.resize(paddedLength); 
    std::vector<float> tmp(paddedLength);
    cudaCompress::util::dwtFloatForwardCPU(tmp.data(), src.data(), src.size(), 0, 0);
    std::copy(tmp.begin(), tmp.begin() + paddedLength / 2, src.begin());
    cudaCompress::util::dwtFloatForwardCPU(src.data(), tmp.data(), tmp.size() / 2, tmp.size() / 2, tmp.size() / 2);
    std::vector<cudaCompress::Symbol16> symbols(src.size());
    cudaCompress::util::quantizeToSymbols(symbols.data(), src.data(), src.size(), quantizationStep);
	cudaCompress::BitStream* arr[]{&bitStream};
    std::vector<cudaCompress::Symbol16>* sArr[]{&symbols};
    cudaCompress::encodeRLHuffCPU(arr, sArr, 1, symbols.size());
    symbolsSize = symbols.size();
}

static void decompressVector(std::vector<uint32_t> src, float quantizationStep, uint32_t symbolsSize, /*out*/ std::vector<float>& data){
    cudaCompress::BitStreamReadOnly bs(src.data(), src.size() * sizeof(src[0]) * 8);
	cudaCompress::BitStreamReadOnly* dec[]{&bs};
	std::vector<cudaCompress::Symbol16> nS(symbolsSize);
	std::vector<cudaCompress::Symbol16>* ss[]{&nS};
	cudaCompress::decodeRLHuffCPU(dec, ss, symbolsSize, 1, symbolsSize);
	std::vector<float> result2(symbolsSize);
    data.resize(symbolsSize);
	cudaCompress::util::unquantizeFromSymbols(data.data(), nS.data(), nS.size(), quantizationStep);
	result2 = data;
	cudaCompress::util::dwtFloatInverseCPU(result2.data(), data.data(), data.size() / 2, data.size() / 2, data.size() / 2);
	cudaCompress::util::dwtFloatInverseCPU(data.data(), result2.data(), data.size());
}

static CachingElementsInfo getCachingElements(const std::string_view& tempPath){
    robin_hood::unordered_map<std::string, std::vector<CacheInfo>> cacheElements;  //stores for each cache id the files in which a bundle is located
    uint32_t levelCount = 0;
    // getting all cache files and extracting cluster from the headerinformation
    for(const auto& entry: std::filesystem::directory_iterator(tempPath)){
        if(entry.is_regular_file() && entry.path().string().find('.') == std::string::npos){    // is cache file
            std::ifstream file(entry.path());
            int cacheBundles; file >> cacheBundles;
            for(int i = 0; i < cacheBundles; ++i){
                std::string id;
                file >> id;
                CacheInfo info;
                info.file = entry.path();
                file >> info.offset >> info.size;
                cacheElements[id].push_back(std::move(info));
                levelCount = std::max(levelCount, static_cast<uint32_t>(std::count(id.begin(), id.end(), '_')));
            }
        }
    }
    return {levelCount, cacheElements};
}

static std::vector<std::vector<int>> getCombinations(int eN, int n){
    // code by https://stackoverflow.com/a/9430993
    std::vector<std::vector<int>> ret;
    if(n > eN)
        throw std::runtime_error{"The elements vector has less elements than the permutation dimension should have"};
    
    std::vector<bool> v(eN, false);
    std::fill(v.begin(), v.begin() + n, true);
    do{
        ret.push_back({});
        for(int i = 0; i < eN; ++i){
            if(v[i]){
                ret.back().push_back(i);
            }
        }
    } while(std::prev_permutation(v.begin(), v.end()));
    return ret;
}
//private functions end ------------------------------------------------------------------------

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
            double epsMult = std::min(pow(1.0/lvlMultiplier, 1.0/dataPoint.size()), .5);
            std::shared_ptr<HierarchyCreateNode> root;
            CompressionMethod compressionMethod{CompressionMethod::HashVectorLeaders};
            switch(compressionMethod){
                case CompressionMethod::Leaders:            root = std::make_shared<LeaderNode>(dataPoint, lvl0eps, epsMult, 0, levels); break;
                case CompressionMethod::VectorLeaders:      root = std::make_shared<VectorLeaderNode>(dataPoint, lvl0eps, epsMult, 0, levels); break;
                case CompressionMethod::Hash:               root = std::make_shared<HashNode>(dataPoint, lvl0eps, epsMult, 0, levels); break;
                case CompressionMethod::HashVectorLeaders:  root = std::make_shared<HashVectorLeaderNode>(dataPoint, lvl0eps, epsMult, 0, levels); break;
                case CompressionMethod::HashLeader:         root = std::make_shared<HashLeaderNode>(dataPoint, lvl0eps, epsMult, 0, levels); break;
            }
            std::shared_ptr<CacheManagerInterface> cacheManager{};
            CachingMethod cachingMethod{CachingMethod::Bundled};
            switch(cachingMethod){
                case CachingMethod::Native: break;
                case CachingMethod::Bundled: cacheManager = std::make_shared<BundlingCacheManager>(std::string(outputFolder) + "/temp"); break;
                case CachingMethod::Single: break;
            }
            std::shared_mutex cacheMutex;                            //mutex for the root node to control insert/cache access

            const int checkInterval = 1000;
            std::atomic<int> sizeCheck = checkInterval;
            auto threadFunc = [](int threadId, DataLoader* loader, HierarchyCreateNode* root, std::shared_mutex* cacheMutex, uint32_t maxMemoryMB, std::string tempPath, std::atomic<int>* sizeCheckk, CacheManagerInterface* cacheManager){
                auto& sizeCheck = *sizeCheckk;
                std::vector<float> threadData;
                while(loader->getNextNormalized(threadData)){
                    //insert into the hirarchy
                    std::shared_lock<std::shared_mutex> insertLock(*cacheMutex);
                    root->addDataPoint(threadData);
                    insertLock.unlock();

                    //should add caching strategies to avoid memory overflow and inbetween writeouts
                    if(--sizeCheck < 0 && threadId == 0){
                        std::unique_lock<std::shared_mutex> lock(*cacheMutex); // locking the root node unique to do caching
                        sizeCheck = checkInterval;
                        size_t structureSize = root->getByteSize();
                        bool doCache = structureSize > size_t(maxMemoryMB) * 1024 * 1024;
                        float get{}, cacheT{}, size{};
                        uint32_t getC{}, cacheC{}, sizeC{};
                        while(doCache && structureSize > size_t(maxMemoryMB) * 1024 * 1024 * .7){   //shrink to 70%
                            HierarchyCreateNode::HierarchyCacheInfo cacheInfo{};
                            cacheInfo.cachingSize = structureSize -  size_t(maxMemoryMB) * 1024 * 1024 * .7;
                            {
                                PCUtil::AverageWatch watch(get, getC);
                                root->getCacheNodes(cacheInfo);
                            }
                            std::vector<float> half(threadData.size(), .5f);
                            {
                                std::set<HierarchyCreateNode*> cacheNodes;
                                while(cacheInfo.queue.size()){
                                    cacheNodes.insert(cacheInfo.queue.top().node);
                                    cacheInfo.queue.pop();
                                }
                                PCUtil::AverageWatch watch(cacheT, cacheC);
                                dynamic_cast<VectorLeaderNode*>(root)->cacheNodes(*cacheManager, "", half.data(), .5f, cacheNodes);
                            }
                            {
                                PCUtil::AverageWatch w(size, sizeC);
                                structureSize = root->getByteSize();
                            }
                        }
                        if(doCache && cacheManager)
                            cacheManager->postDataInsert();
                    }
                }
            };
            {
                std::cout << "Creating all threds" << std::endl;
                std::vector<std::future<void>> threads(amtOfThreads);
                for(int i = 0; i < amtOfThreads; ++i){
                    threads[i] = std::async(threadFunc, i, loader, root.get(), &cacheMutex, maxMemoryMB, tempPath, &sizeCheck, cacheManager.get());
                }
                std::cout << "Threads up and running" << std::endl;
            }   // all futures are automatically joined at the end of the section
            std::cout << "Threads done with hierarchy creation" << std::endl;
            
            //final writeout to disk
            bool hellYeah = true;
            std::vector<float> half(dataPoint.size(), .5f);
            if(cacheManager){
                dynamic_cast<VectorLeaderNode*>(root.get())->cacheNodes(*cacheManager, "", half.data(), .5f, {});
                cacheManager->postDataInsert();
                cacheManager->finish();
            }
            else
                root->cacheNode(tempPath, "", half.data(), .5f, root.get());

            //info file containing caching strategy and attributes
            std::ofstream file(std::string(outputFolder) + "/attr.info", std::ios_base::binary);
            file << CachingMethodNames[static_cast<int>(cachingMethod)] << "\n";
            std::vector<Attribute> attributes;
            size_t tmp;
            loader->dataAnalysis(tmp, attributes);
            for(auto& a: attributes){
                file << a.name << " " << a.min << " " << a.max << "\n"; 
            }
            std::cout << "Hierarchy writeout done" << std::endl;
        }
        catch(std::filesystem::filesystem_error err){
            std::cout << "Error trying to open output folder " << err.path1() << " with code: " << err.code() << std::endl;
        }
    }
    
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep) 
    {
        //if other types of caching was used automatically adopt to other strategies
        std::ifstream info(std::string(outputFolder) + "/attr.info");
        std::string cachingMethod; info >> cachingMethod;
        if(cachingMethod == CachingMethodNames[static_cast<int>(CachingMethod::Bundled)])
            return compressBundledTempHierarchy(outputFolder, amtOfThreads, quantizationStep);
        std::string tempPath = std::string(outputFolder) + "/temp";
        std::vector<std::string> cacheFiles;
        // getting all cache files
        for(const auto& entry: std::filesystem::directory_iterator(tempPath)){
            if(entry.is_regular_file() && entry.path().string().find('.') == std::string::npos){    // is cache file
                cacheFiles.push_back(entry.path().string());
            }
        }

        // compressing the cache files
        auto compressThread = [](const std::vector<std::string>& files, const std::string& outputFolder, float quantizationStep){
            for(auto& f: files){
                std::ifstream fs(f, std::ios_base::binary);
                std::vector<float> data;
                int rowLength;
                float eps;
                int dataSize;
                while(fs >> rowLength >> dataSize >> eps){
                    fs.get();   //newline char
                    //reading the data
                    std::vector<float> d(dataSize);
                    fs.read(reinterpret_cast<char*>(d.data()), dataSize * sizeof(d[0]));
                    data.insert(data.end(), d.begin(), d.end());
                    fs.get();   //newline char
                }
                //copying the first datarow as it is the center of the group
                std::vector<float> center(data.begin(), data.begin() + rowLength);
                //converting from row major to column major
                std::vector<float> col(data.size());
                uint32_t colInd = 0;
                for(uint32_t curInd = 0; colInd < data.size(); curInd += rowLength){
                    if(curInd >= data.size()) ++curInd %= data.size();
                    col[colInd++] = data[curInd];
                }
                //compressing the data with 2 dwts, followed by run-length and huffman encoding of quantized symbols
                //padding to size % 4 size
                uint32_t originalLength = col.size();
                uint32_t paddedLength = ((4 - (col.size() & 0b11)) & 0b11) + col.size();
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
                out << rowLength << " " << bitStream.getRawSizeBytes() << " " << symbols.size() << " " << originalLength << " " << quantizationStep << " " << eps << "\n";
                for(int i = 0; i < center.size(); ++i){
                    out << center[i];
                    if(i != center.size() - 1)
                        out << " ";
                }
                out << "\n";
                out.write(reinterpret_cast<char*>(bitStream.getRaw()), bitStream.getRawSizeBytes());
            }
            std::cout << "Thread done compressing and merging" << std::endl;
        };

        {
            std::vector<std::future<void>> futures(amtOfThreads);
            std::cout << "Starging Tmp compression" << std::endl;
            auto curStart = cacheFiles.begin();
            for(int i = 0; i < amtOfThreads; ++i){
                auto curEnd = cacheFiles.begin() + (i + 1) * cacheFiles.size() / amtOfThreads;
                std::vector<std::string> subSet(curStart, curEnd);
                curStart = curEnd;
                futures[i] = std::async(compressThread, subSet, std::string(outputFolder), quantizationStep);
            }
        }
        std::cout << "Tmp compression done" << std::endl;
    }

    void convertTempHierarchy(const std::string_view& outputFolder, int amtOfThreads){
        //adopt to different types of strategies
        std::ifstream info(std::string(outputFolder) + "/attr.info");
        std::string cachingMethod; info >> cachingMethod;
        std::string tempFolder(outputFolder); tempFolder += "/temp";
        if(cachingMethod == CachingMethodNames[static_cast<int>(CachingMethod::Bundled)]){
            auto [levelCount, cacheElements] = getCachingElements(tempFolder);
            using robin_hood_iter = robin_hood::unordered_map<std::string, std::vector<CacheInfo>>::iterator;
            auto compressThread = [](robin_hood::unordered_map<std::string, std::vector<CacheInfo>>* cacheElements, robin_hood_iter begin, robin_hood_iter end, const std::string& outputFolder, std::vector<MutexPair>* mutexes){
                auto& mutes = *mutexes;
                for(;begin != end; ++begin){
                    const auto& id = begin->first;
                    std::vector<float> data;
                    //combining the data fragments from all files for the cluster file
                    uint32_t rowLength; //needed for column to row major conversion
                    float eps;          //needed for writeout information
                    std::vector<float> center;
                    for(auto& file: (*cacheElements)[id]){
                        std::size_t start = file.offset; size_t size = file.size;
                        std::ifstream bundleInfo(file.file);
                        std::string bin(size, ' ');
                        bundleInfo.seekg(start);            //going to the start of the sequence
                        bundleInfo.read(bin.data(), size);  //reading out the data
                        std::stringstream curFragment;
                        curFragment.str(bin);               //setting the streambuffer to the binary data
                        uint32_t dataSize;
                        curFragment >> rowLength >> dataSize >> eps;
                        curFragment.get();   //newline char
                        //reading the data
                        data.resize(data.size() + dataSize);
                        auto insertPointer = &*(data.end() - dataSize);
                        curFragment.read(reinterpret_cast<char*>(insertPointer), dataSize * sizeof(data[0]));
                    }

                    center = std::vector<float>(data.begin(), data.begin() + rowLength);
                    //having all data in row major order inside the data vector -> reorder to column major format
                    std::vector<float> col;
                    row2Col(data, rowLength, col);

                    //skipping compression and instantly writing back data
                    {
                        //appending the data to the correct file
                        uint32_t level = std::count(id.begin(), id.end(), '_') - 1;
                        std::unique_lock<std::mutex> infoLock(mutes[level].info);        //exclusively locking the level files
                        std::unique_lock<std::mutex> dataLock(mutes[level].data);

                        //writing the infos
                        std::string outNameBase = std::string(outputFolder) + "/UnCompLevel" + std::to_string(level);
                        std::ofstream infos(outNameBase + ".info", std::ios_base::app | std::ios_base::binary);
                        std::ofstream dataFile(outNameBase, std::ios_base::app | std::ios_base::binary);
                        uint32_t offset = dataFile.tellp();
                        infos << rowLength << " " << offset << " " << col.size() << " " << col.size() << " " << data.size() << " " << 0 << " " << eps << "\n";
                        for(int i = 0; i < center.size(); ++i){
                            infos << center[i];
                            if(i != center.size() - 1)
                                infos << " ";
                        }
                        infos << "\n";
                        infos.close();                                                  //closing the info file to release the lock
                        infoLock.unlock();                                              //unlocking infos after they have been written to allow following processes to instantly also update the info
                        //writing the compressed data;
                        dataFile.write(reinterpret_cast<char*>(col.data()), col.size() * sizeof(col[0]));
                        dataFile.close();                                               //manually closing the data file to avoid unlock errors with the mutex
                    }
                }
            };
            {
                std::vector<MutexPair> levelMutexes(levelCount);
                std::vector<std::future<void>> futures(amtOfThreads);
                std::cout << "compression::convertTempHierarchy(): Starting conversion" << std::endl;
                auto curStart = cacheElements.begin();
                for(int i = 0; i < amtOfThreads; ++i){
                    auto curEnd = cacheElements.begin();
                    std::advance(curEnd, (i + 1) * cacheElements.size() / amtOfThreads);
                    futures[i] = std::async(compressThread, &cacheElements, curStart, curEnd, std::string(outputFolder), &levelMutexes);
                    curStart = curEnd;
                }
            }
        }
        else{
            std::cout << "compression::convertTempHierarchy(): conversion for caching method " << cachingMethod << " not implemented! Aborting..." << std::endl;
            return;
        }
        
        std::cout << "compression::convertTempHierarchy(): done" << std::endl;
    }

    void compressBundledTempHierarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep){
        std::string tempPath = std::string(outputFolder) + "/temp";
        robin_hood::unordered_map<std::string, std::vector<CacheInfo>> cacheElements;  //stores for each cache id the files in which a bundle is located
        uint32_t levelCount = 0;
        // getting all cache files and extracting cluster from the headerinformation
        for(const auto& entry: std::filesystem::directory_iterator(tempPath)){
            if(entry.is_regular_file() && entry.path().string().find('.') == std::string::npos){    // is cache file
                std::ifstream file(entry.path());
                int cacheBundles; file >> cacheBundles;
                for(int i = 0; i < cacheBundles; ++i){
                    std::string id;
                    file >> id;
                    CacheInfo info;
                    info.file = entry.path();
                    file >> info.offset >> info.size;
                    cacheElements[id].push_back(std::move(info));
                    levelCount = std::max(levelCount, static_cast<uint32_t>(std::count(id.begin(), id.end(), '_')));
                }
            }
        }
        assert(levelCount < 5); //debugassert

        auto compressThread = [](robin_hood::unordered_map<std::string, std::vector<CacheInfo>>* cacheElements, std::vector<std::string> compressElements, const std::string& outputFolder, float quantizationStep, std::vector<MutexPair>* mutexes){
            auto& mutes = *mutexes;
            for(const auto& id: compressElements){
                std::vector<float> data;
                //combining the data fragments from all files for the cluster file
                uint32_t rowLength; //needed for column to row major conversion
                float eps;          //needed for writeout information
                cudaCompress::BitStream compressed;
                uint32_t symbolsSize;
                std::vector<float> center;
                for(auto& file: (*cacheElements)[id]){
                    std::size_t start = file.offset; size_t size = file.size;
                    std::ifstream bundleInfo(file.file);
                    std::string bin(size, ' ');
                    bundleInfo.seekg(start);            //going to the start of the sequence
                    bundleInfo.read(bin.data(), size);  //reading out the data
                    std::stringstream curFragment;
                    curFragment.str(bin);               //setting the streambuffer to the binary data
                    uint32_t dataSize;
                    curFragment >> rowLength >> dataSize >> eps;
                    curFragment.get();   //newline char
                    //reading the data
                    data.resize(data.size() + dataSize);
                    auto insertPointer = &*(data.end() - dataSize);
                    curFragment.read(reinterpret_cast<char*>(insertPointer), dataSize * sizeof(data[0]));
                }

                center = std::vector<float>(data.begin(), data.begin() + rowLength);
                //having all data in row major order inside the data vector -> reorder to column major format
                std::vector<float> col;
                row2Col(data, rowLength, col);
                compressVector(col, quantizationStep, compressed, symbolsSize);

                {
                    //appending the data to the correct file
                    uint32_t level = std::count(id.begin(), id.end(), '_') - 1;
                    std::unique_lock<std::mutex> infoLock(mutes[level].info);        //exclusively locking the level files
                    std::unique_lock<std::mutex> dataLock(mutes[level].data);

                    //writing the infos
                    std::string outNameBase = std::string(outputFolder) + "/level" + std::to_string(level);
                    std::ofstream infos(outNameBase + ".info", std::ios_base::app | std::ios_base::binary);
                    std::ofstream dataFile(outNameBase, std::ios_base::app | std::ios_base::binary);
                    uint32_t offset = dataFile.tellp();
                    infos << rowLength << " " << offset << " " << compressed.getRawSizeBytes() << " " << symbolsSize << " " << data.size() << " " << quantizationStep << " " << eps << "\n";
                    for(int i = 0; i < center.size(); ++i){
                        infos << center[i];
                        if(i != center.size() - 1)
                            infos << " ";
                    }
                    infos << "\n";
                    infos.close();                                                  //closing the info file to release the lock
                    infoLock.unlock();                                              //unlocking infos after they have been written to allow following processes to instantly also update the info
                    //writing the compressed data;
                    dataFile.write(reinterpret_cast<char*>(compressed.getRaw()), compressed.getRawSizeBytes());
                    dataFile.close();                                               //manually closing the data file to avoid unlock errors with the mutex
                }
            }
        };
        {
            std::vector<MutexPair> levelMutexes(levelCount);
            std::vector<std::future<void>> futures(amtOfThreads);
            std::cout << "Starging Tmp compression" << std::endl;
            auto curStart = cacheElements.begin();
            for(int i = 0; i < amtOfThreads; ++i){
                auto curEnd = cacheElements.begin();
                std::advance(curEnd, (i + 1) * cacheElements.size() / amtOfThreads);
                std::vector<std::string> subSet;
                while(curStart != curEnd)
                    subSet.push_back(curStart++->first);
                futures[i] = std::async(compressThread, &cacheElements, subSet, std::string(outputFolder), quantizationStep, &levelMutexes);
            }
        }
        std::cout << "Tmp compression done" << std::endl;
    }
    
    void loadAndDecompress(const std::string_view& file, Data& data) 
    {
	    std::ifstream in(file.data());
	    uint32_t colCount, byteSize, symbolsSize, dataSize;
	    float quantizationStep, eps;
	    in >> colCount >> byteSize >> symbolsSize >> dataSize >> quantizationStep >> eps;
	    in.get();	//skipping newline
        float t;
        for(int i = 0; i < colCount; ++i)
            in >> t;
        in.get();   //skipping newline
	    std::vector<uint32_t> bytes(byteSize / 4);
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

    void loadAndDecompressBundled(const std::string_view& levelFile, size_t offset, Data& data){
        auto extensionPos = levelFile.find(".info");
        std::string dataFile(levelFile.substr(0, extensionPos));

        //reading infos from the infos file
        std::string levlFileC(levelFile);
        if(extensionPos == std::string::npos) levlFileC += ".info";
        std::ifstream info(levlFileC, std::ios_base::binary);
        info.seekg(offset);
        size_t rowLength, dataOffset, compressedByteSize, symbolSize, dataSize;
        float quantizationStep, eps;
        info >> rowLength >> dataOffset >> compressedByteSize >> symbolSize >> dataSize >> quantizationStep >> eps;
        info.close();

        //loading and decompressing the data
        std::ifstream dataF(dataFile, std::ios_base::binary);
        std::vector<uint32_t> bytes(compressedByteSize / sizeof(uint32_t));
        dataF.seekg(dataOffset);
        dataF.read(reinterpret_cast<char*>(bytes.data()), compressedByteSize);
        std::vector<float> decompressedData;
        decompressVector(bytes, quantizationStep, symbolSize, decompressedData);
        
        //assigning the data to the data object coming in
        data.columns.resize(rowLength);
        data.columnDimensions.resize(rowLength, {0});
        uint32_t colSize = dataSize / rowLength;
        data.dimensionSizes = {colSize};
        for(int i = 0; i < rowLength; ++i){
            data.columns[i] = std::vector<float>(decompressedData.begin() + i * colSize, decompressedData.begin() + (i + 1) * colSize);
        }
    }

    void combineData(std::vector<Data>& data, Data& dst){
        dst.columnDimensions = data[0].columnDimensions;
        dst.dimensionSizes = {0};
        dst.columns.clear(); dst.columns.resize(data[0].columns.size());
        for(int i = 0; i < data.size(); ++i){
            dst.dimensionSizes[0] += data[i].dimensionSizes[0];
            for(int c = 0; c < data[i].columns.size(); ++c){
                dst.columns[c].insert(dst.columns[c].begin(), data[i].columns[c].begin(), data[i].columns[c].end());
            }
        }
    }

    void createNDHierarchy(const std::string_view& outputFolder, DataLoader* loader, CachingMethod cachingMethod, int startCluster, int clusterMultiplikator, int dimensionality, int levels, int maxMb, int amtOfThreads){
        //Cluster indices are stored for teh cluster counts, so max number of clusters has to be smaller than max uint16_t to be able to store the things
        assert(startCluster * pow(clusterMultiplikator, levels - 1) < std::numeric_limits<uint16_t>::max());
        size_t dataSize;
        std::vector<Attribute> attributes;
        loader->dataAnalysis(dataSize, attributes);
        const auto combinationIndices = getCombinations(attributes.size(), dimensionality);
        std::cout << "Amount of dimension combinations: " << combinationIndices.size() << std::endl;
        //for(const auto& comb: combinationIndices){
        //    std::cout << "[";
        //    for(int i: comb){
        //        std::cout << i << ",";
        //    }
        //    std::cout << "\b]" << std::endl;
        //}
        struct ShortVecHasher{
            size_t operator()(const std::vector<uint16_t>& v)const{
                size_t hash =v.size();
                for(const auto& k: v){
                    hash ^= k + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                return hash;
            }
        };
        std::vector<std::vector<robin_hood::unordered_map<uint32_t, uint32_t>>> attributeHierarchyCenters(attributes.size(), std::vector<robin_hood::unordered_map<uint32_t, uint32_t>>(levels));
        using ClusterCounts = std::vector<robin_hood::unordered_map<std::vector<uint16_t>, uint32_t, ShortVecHasher>> ;
        ClusterCounts clusterCounts(combinationIndices.size());   // for each attribute combination one clustering exists
        std::vector<uint32_t> clusterLevelSizes(levels);
        for(int l = 0; l < levels; ++l)
            clusterLevelSizes[l] = startCluster * pow(clusterMultiplikator, l);
        
        std::vector<CenterData> data;                    //data contains all data points for attributeHierarchyCreationCenters

        std::vector<float> p;
        loader->reset();

        std::vector<std::pair<int, int>> bounds;
        std::vector<std::thread> clusterThreads(amtOfThreads - 1);
        int mainEnd = combinationIndices.size() / amtOfThreads;
        int prev = mainEnd;
        for(int i = 1; i < amtOfThreads; ++i){
            int start = prev;
            int end = combinationIndices.size() * (i + 1) / amtOfThreads;
            prev = end;
            bounds.push_back({start, end});
        }
        std::vector<PCUtil::Semaphore> threadStartSemaphores(amtOfThreads - 1);
        std::vector<PCUtil::Semaphore> threadFinishedSemaphores(amtOfThreads - 1);
        std::atomic_bool done{false};
        auto threadEx = [&](const std::vector<float>* p, std::pair<int,int> ind, const std::vector<std::vector<int>>* combinations, ClusterCounts* clusterCounts, int baseClusterSize, int threadInd){
            while(!done){
                threadStartSemaphores[threadInd].acquire();    //wait for main thread
                if(done)
                    break;
                int dimensionality = (*combinations)[0].size();
                    for(int c = ind.first; c < ind.second; ++c){
                    std::vector<uint16_t> k(dimensionality);
                    for(int d = 0; d < dimensionality; ++d){
                        int a = (*combinations)[c][d];
                        k[d] = (*p)[a] * baseClusterSize;
                    }
                    ++((*clusterCounts)[c][k]);
                }
                threadFinishedSemaphores[threadInd].release();  //signaling finish
            }
        };
        for(int t = 0; t < clusterThreads.size(); ++t){
            clusterThreads[t] = std::thread(threadEx, &p, bounds[t], &combinationIndices, &clusterCounts, clusterLevelSizes.back(), t);
        }

        uint32_t centerAdoptionC, clusteringC;
        float centerAdoptionT, clusteringT;
        while(loader->getNextNormalized(p)){
            for(auto& s: threadStartSemaphores)  //running clustering threads
                s.release();
            //inserting into the hierarchy levels
            {
            //PCUtil::AverageWatch centerAd(centerAdoptionT, centerAdoptionC);
            for(int a = 0; a < attributes.size(); ++a){
                float axisValue = p[a];
                for(int l = 0; l < levels; ++l){
                    float clustAmt = clusterLevelSizes[l];
                    uint32_t childId = axisValue * clustAmt;
                    if(attributeHierarchyCenters[a][l].contains(childId)){
                        CenterData& cd = data[attributeHierarchyCenters[a][l][childId]];
                        float a = float(cd.count) / float(++cd.count);
                        cd.val = a * cd.val + (1. - a) * axisValue;
                        if(axisValue < cd.min)
                            cd.min = axisValue;
                        if(axisValue > cd.max)
                            cd.max = axisValue;
                    }
                    else{
                        attributeHierarchyCenters[a][l][childId] = data.size();
                        data.push_back({axisValue, axisValue, axisValue, 1});
                    }
                }
            }
            }

            // doing clustering
            // clustering is only done on the lowest level. The higher levels are then abstracted on compression
            {
            //PCUtil::AverageWatch clusterWatch(clusteringT, clusteringC);
            for(int c = 0; c < mainEnd; ++c){
                std::vector<uint16_t> k(dimensionality);
                for(int d = 0; d < dimensionality; ++d){
                    int a = combinationIndices[c][d];
                    k[d] = p[a] * clusterLevelSizes.back();
                }
                ++clusterCounts[c][k];
            }
            //waiting clustering threads
            for(int s = 0; s < threadFinishedSemaphores.size(); ++s)
                threadFinishedSemaphores[s].acquire();
            }
        }
        done = true;
        for(auto& s: threadStartSemaphores)
            s.release();
        for(auto& t: clusterThreads)
            t.join();

        // ----------------------------------------------------------------------------------------------
        // writeout of the attribute Hierarchy
        // ----------------------------------------------------------------------------------------------
        std::ofstream acFile(std::string(outputFolder) + "/attributeCenters.ac", std::ios_base::binary);
        //writing header information
        //header information contains the offsets for each attribute centers at each hierarchy levels
        //specifically the hierarchy contains the following iformation
        //  info: [[uint, uint] offset, size] for each attribute the list of offsets and sizes is placed in the beginning
        //  data: [[float, float, float, uint] center, min, max, count] all centers and their counts are then put in sequence
        std::vector<char> writeBuffer;
        std::vector<uint32_t> headerInfo;
        uint32_t offset = attributeHierarchyCenters.size() * attributeHierarchyCenters.front().size() * 2 * sizeof(uint32_t);
        for(const auto& att: attributeHierarchyCenters){
            for(const auto& lvl: att){
                uint32_t size = lvl.size() * sizeof(CenterData);
                headerInfo.push_back(offset);
                headerInfo.push_back(size);
                offset += size;
            }
        }
        acFile.write(reinterpret_cast<char*>(headerInfo.data()), headerInfo.size() * sizeof(headerInfo[0]));
        //writing data information
        for(const auto& att: attributeHierarchyCenters){
            for (const auto& lvl: att){
                std::vector<CenterData> linData(lvl.size());
                uint32_t ind{0};
                for(auto [id, i]: lvl){
                    linData[ind++] = data[i];
                }
                acFile.write(reinterpret_cast<char*>(linData.data()), linData.size() * sizeof(linData[0]));
            }
        }
        acFile.close();     // closing early to safe ram

        // ----------------------------------------------------------------------------------------------
        // writeout of the cluster information 
        // ----------------------------------------------------------------------------------------------
        std::ofstream combinationInfo(std::string(outputFolder) + "/combination.info", std::ios_base::binary);
        for(const auto& comb: combinationIndices){
            for(auto ind: comb)
                combinationInfo << ind << " ";
            combinationInfo << "\n";
        }
        combinationInfo.close();

        std::ofstream clusterData(std::string(outputFolder) + "/cluster.cd", std::ios_base::binary);
        // writing header information for cluster data
        // header information contains the offsets for each index-combination found in the combanitaoin.info file
        // specifically the cluster fiel is built up as
        //  [info<cluster1>, info<cluster2>, ..., info<cluster3>][data<cluster1>, data<cluster2>, ..., data<cluster3>]
        //  info: [uint, uint] offset and size in bytes for each block of index combination
        //  data: [vector<uint16_t>, uint] variable length cluster id (length is padded to 32 bit dimensionality) uint for cluster counts
        uint32_t paddedIndexSize = (combinationIndices[0].size() + 1) >> 1;
        offset = combinationIndices.size() * 2 * sizeof(uint32_t);
        headerInfo.clear();
        for(int comb = 0; comb < combinationIndices.size(); ++comb){
            uint32_t size = clusterCounts[comb].size() * (paddedIndexSize + 1) * sizeof(uint32_t);
            headerInfo.push_back(offset);
            headerInfo.push_back(size);
            offset += size;
        }
        clusterData.write(reinterpret_cast<char*>(headerInfo.data()), headerInfo.size() * sizeof(headerInfo[0]));
        assert(clusterData.tellp() == combinationIndices.size() * 2 * sizeof(uint32_t));
        // writing data information
        for(const auto& comb: clusterCounts){
            std::vector<uint32_t> clusterWrite{};
            for(const auto& [id, count]: comb){
                std::vector<uint16_t> idc(id.begin(), id.end());
                idc.resize(paddedIndexSize * 2, 0);  //filling with 0
                clusterWrite.insert(clusterWrite.end(), reinterpret_cast<uint32_t*>(idc.data()), reinterpret_cast<uint32_t*>(idc.data()) + paddedIndexSize);
                clusterWrite.push_back(count);
            }
            clusterData.write(reinterpret_cast<char*>(clusterWrite.data()), clusterWrite.size() * sizeof(clusterWrite[0]));
        }
        assert(clusterData.tellp() == offset);  //checking if the counting from before is correct 
        clusterData.close();

        //info file containing caching strategy and attributes
        std::ofstream file(std::string(outputFolder) + "/attr.info", std::ios_base::binary);
        file << CachingMethodNames[static_cast<int>(cachingMethod)] << "\n";
        for(auto& a: attributes){
            file << a.name << " " << a.min << " " << a.max << "\n"; 
        }
        std::cout << "Donesen!" << std::endl;
    }

    void convertNDHierarchy(const std::string_view& outputFolder, int amtOfThreads){
        
    }
}