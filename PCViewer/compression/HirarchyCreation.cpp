#include "HirarchyCreation.hpp"

#include "LeaderNode.hpp"
#include "../rTree/RTreeDynamic.h"
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
    void createHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads) 
    {
        createTempHirarchy(outputFolder, loader, lvl0eps, levels, lvlMultiplier, maxMemoryMB, amtOfThreads);
        compressTempHirarchy(outputFolder, amtOfThreads);
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
            std::ofstream file(tempPath + "/attr.info", std::ios_base::binary);
            file.clear();
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
    
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads) 
    {
        
    }
}