#include "HierarchyImportManager.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "HirarchyCreation.hpp"

HierarchyImportManager::HierarchyImportManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines) :
_maxLines(maxDrawLines), _hierarchyFolder(hierarchyFolder)
{
    // loading all hierarchy files
    for(const auto& entry: std::filesystem::directory_iterator(hierarchyFolder)){
        if(entry.is_regular_file()){
            if(!entry.path().has_extension())   //standard compressed hierarchy file
                _hierarchyFiles.push_back(entry.path().string());
            else if(entry.path().extension().string() == "info"){ //configuration file containing column information
                std::ifstream info(entry.path());
                int colCount = 0;
                bool foundReserverdAttribute = false;
                while(!info.eof() && info.good()){
                    Attribute a;
                    info >> a.name >> a.min >> a.max;
                    a.originalName = a.name;
                    info.get(); //skip newline
                    bool reserverdAttribute = std::find(compressionConstants::reservedAttributeNames.begin(), compressionConstants::reservedAttributeNames.end(), a.name) != compressionConstants::reservedAttributeNames.end();
                    foundReserverdAttribute |= reserverdAttribute;
                    if(foundReserverdAttribute && !reserverdAttribute){
                        std::cout << "Attribute mangling in the compressed hierarchy. Nothing loaded" << std::endl;
                        _hierarchyValid = false;
                        return;
                    }
                    if(reserverdAttribute)
                        _reservedAttributes.push_back(a);
                    else
                        _attributes.push_back(a);
                }
            }
        }
    }
    // starting to find the base layer (last hierarchy layer with less than a million lines)
    uint32_t maxDepth = 0;
    std::vector<uint32_t> levelLineCount;
    uint32_t columnAmt = _reservedAttributes.size() + _attributes.size();
    for(auto& s: _hierarchyFiles){
        uint32_t curPos = s.size() - 1;
        uint32_t hierarchyDepth = 0;
        while(curPos >= 0 && s[curPos] != '/' && s[curPos] != '\\'){
            if(s[curPos] == '_') ++hierarchyDepth;
            --curPos;
        }
        --hierarchyDepth;
        if(hierarchyDepth > maxDepth) maxDepth = hierarchyDepth;
        levelLineCount.resize(maxDepth, 0);
        //loading the file header and getting the data point sizes
        std::ifstream f(s, std::ios_base::binary);
        uint byteSize, symbolsSize, dataSize;
	    float quantizationStep;
	    f >> byteSize >> symbolsSize >> dataSize >> quantizationStep;
        f.close();
        levelLineCount[hierarchyDepth] += dataSize / columnAmt;
        _levelFiles.resize(maxDepth);
        _levelFiles[hierarchyDepth].push_back(std::string_view(s));
    }

    //setting the base level
    for(int i = 0; i < levelLineCount.size(); ++i){
        if(levelLineCount[i] < _maxLines)
            _baseLevel = i;
    }

    openHierarchyFiles(_levelFiles[_baseLevel]);
}

void HierarchyImportManager::openHierarchyFiles(const std::vector<std::string_view>& files){
    using namespace std::chrono_literals;
    // if loading is still active return and dont open new hierarchy files
    if(_dataLoadFuture.valid() && _dataLoadFuture.wait_for(0s) == std::future_status::timeout) 
        return;
    
    auto exec = [&](std::vector<std::string_view> files){
        std::vector<Data> dataVec(files.size());
        uint32_t i = 0;
        for(auto& f: files){
            compression::loadAndDecompress(f, dataVec[i++]);
        }
        compression::combineData(dataVec, _nextData);
        newDataLoaded = true;
    };

    _dataLoadFuture = std::async(exec, files);
}
