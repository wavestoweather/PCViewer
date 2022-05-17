#define NOSTATICS
#include "HierarchyImportManager.hpp"
#undef NOSTATICS
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>

HierarchyImportManager::HierarchyImportManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines) :
_maxLines(maxDrawLines), _hierarchyFolder(hierarchyFolder)
{
    // loading all hierarchy files
    bool foundInfoFile{false};
    for(const auto& entry: std::filesystem::directory_iterator(hierarchyFolder)){
        if(entry.is_regular_file()){
            if(!entry.path().has_extension() || entry.path().filename().string().find("level") != std::string::npos)   //standard compressed hierarchy file
                _hierarchyFiles.push_back(entry.path().string());
            else if(entry.path().filename().string() == "attr.info"){ //configuration file containing column information
                foundInfoFile = true;
                std::ifstream info(entry.path());
                std::string cachingMethod;
                info >> cachingMethod;
                _cachingMethod = static_cast<compression::CachingMethod>(std::find(compression::CachingMethodNames, compression::CachingMethodNames + static_cast<int>(compression::CachingMethod::MethodCount), cachingMethod) - compression::CachingMethodNames);
                if(_cachingMethod == compression::CachingMethod::MethodCount){
                    std::cout << "HierarchyImportManager::HierarchyImportManager(): Not known compression method \"" << cachingMethod << "\" encountered. Nothing loaded" << std::endl;
                    _hierarchyFiles.clear();
                    return;
                }
                int colCount = 0;
                bool foundReserverdAttribute = false;
                while(!info.eof() && info.good()){
                    Attribute a;
                    info >> a.name >> a.min >> a.max;
                    a.originalName = a.name;
                    info.get(); //skip newline
                    bool reserverdAttribute = false; // std::find(compressionConstants::reservedAttributeNames.begin(), compressionConstants::reservedAttributeNames.end(), a.name) != compressionConstants::reservedAttributeNames.end();
                    foundReserverdAttribute |= reserverdAttribute;
                    if(foundReserverdAttribute && !reserverdAttribute){
                        std::cout << "Attribute mangling in the compressed hierarchy. Nothing loaded" << std::endl;
                        _hierarchyValid = false;
                        return;
                    }
                    if(reserverdAttribute)
                        _reservedAttributes.push_back(a);
                    else if(a.name.size())
                        _attributes.push_back(a);
                }
            }
        }
    }
    if(!foundInfoFile){
        std::cout << "HierarchyImportManager::HierarchyImportManager(): There was no .info file found in the given directory. Nothing loaded" << std::endl;
        _hierarchyFiles.clear();
        return;
    }
    _reservedAttributes.push_back({});
    // starting to find the base layer (last hierarchy layer with less than a million lines)
    std::vector<std::vector<size_t>> bundleOffsets;
    switch(_cachingMethod){
    case compression::CachingMethod::Bundled:{
        for(auto& str: _hierarchyFiles){
            if(size_t p = str.find(".info"); p != std::string::npos){
                size_t levelEnd = str.find("level") + std::string("level").size();
                int lvl = atoi(str.substr(levelEnd, p).c_str());
                _levelInfos.resize(lvl + 1);
                _levelInfos[lvl].push_back(str);
            }
            else{
                size_t levelEnd = str.find("level") + std::string("level").size();
                int lvl = atoi(str.substr(levelEnd, p).c_str());
                _levelFiles.resize(lvl + 1);
                _levelFiles[lvl].push_back(str);
            }
        }
        //getting the best base level
        for(auto& levelInfo: _levelInfos){
            std::vector<std::vector<size_t>> curOffsets{{0}};
            std::ifstream file(std::string(levelInfo[0]), std::ios_base::binary);
            uint32_t curLineCount = 0;
            uint32_t rowLength, offset, byteSize, symbolSize, dataSize;
            float quantizationStep, eps;
            while(file >> rowLength >> offset >> byteSize >> symbolSize >> dataSize >> quantizationStep >> eps){
                // ignoring the center values
                for(int a = 0; a < rowLength; ++a) file >> eps;
                curLineCount += dataSize / rowLength;
                if(curLineCount > _maxLines)
                    break;
                file.get(); //skipping newline for security
                curOffsets[0].push_back(file.tellg());
            }
            curOffsets[0].pop_back();
            if(curLineCount < _maxLines){
                bundleOffsets = curOffsets;
                ++_baseLevel;
            }
            else{
                if(_baseLevel == 0)
                    bundleOffsets = curOffsets;
                break;
            }
        }
        if(_baseLevel > 0)
            --_baseLevel;
        break;
    }
    default:{
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
            if(hierarchyDepth > maxDepth) maxDepth = hierarchyDepth;
            hierarchyDepth--;
            levelLineCount.resize(maxDepth, 0);
            //loading the file header and getting the data point sizes
            std::ifstream f(s, std::ios_base::binary);
            uint32_t colCount, byteSize, symbolsSize, dataSize;
	        float quantizationStep, eps;
	        f >> colCount >> byteSize >> symbolsSize >> dataSize >> quantizationStep >> eps;
            f.close();
            levelLineCount[hierarchyDepth] += dataSize / colCount;
            _levelFiles.resize(maxDepth);
            _levelFiles[hierarchyDepth].push_back(std::string_view(s));
        }

        //setting the base level
        for(int i = 0; i < levelLineCount.size(); ++i){
            if(levelLineCount[i] < _maxLines)
                _baseLevel = i;
        }
    }
    }

    switch(_cachingMethod){
    case compression::CachingMethod::Bundled:
        openHierarchyFiles(_levelInfos[_baseLevel], bundleOffsets);
        break;
    default:
        openHierarchyFiles(_levelFiles[_baseLevel]);
    }
}

void HierarchyImportManager::notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes) 
{
    std::cout << "HierarchyImportManager::notifyBrushUpdate(): updating..." << std::endl;
    if(_cachingMethod != compression::CachingMethod::Bundled)
        return;
    
    //converting the brushes to local coordinates (the normalized coordinates)
    std::vector<RangeBrush> normalizedBrushes;
    for(auto& b: rangeBrushes){
        RangeBrush rb{};
        for(auto& r: b){
            brushing::AxisRange ar{};
            ar.axis = r.axis;
            double diff = _attributes[r.axis].max - _attributes[r.axis].min;
            ar.min = (r.min - _attributes[r.axis].min) / diff;
            ar.max = (r.max - _attributes[r.axis].min) / diff;
            rb.push_back(ar);
        }
        normalizedBrushes.push_back(rb);
    }
    Polygons normalizedLassos;
    for(auto& l: lassoBrushes){
        Polygon p{};
        p.attr1 = l.attr1;
        p.attr2 = l.attr2;
        for(auto& bp: l.borderPoints){
            ImVec2 v{};
            v.x = bp.x - _attributes[l.attr1].min / (_attributes[l.attr1].max - _attributes[l.attr1].min);
            v.y = bp.y - _attributes[l.attr2].min / (_attributes[l.attr2].max - _attributes[l.attr2].min);
            p.borderPoints.push_back(v);
        }
        normalizedLassos.push_back(p);
    }

    //checking if something changed from last notifyBrushUpdate() call
    bool change = false;
    if(normalizedBrushes.size() != _curRangeBrushes.size())
        change = true;
    for(int i = 0; !change && i < normalizedBrushes.size(); ++i){
        if(normalizedBrushes[i].size() != _curRangeBrushes[i].size()){
            change = true;
            break;
        }
        for(int j = 0;!change && j < normalizedBrushes[i].size(); ++j){
            if(normalizedBrushes[i][j].axis != _curRangeBrushes[i][j].axis ||
                normalizedBrushes[i][j].min != _curRangeBrushes[i][j].min ||
                normalizedBrushes[i][j].max != _curRangeBrushes[i][j].max)
                change = true;
        }
    }
    if(normalizedLassos.size() != _curLassoBrushes.size())
        change = true;
    for(int i = 0;!change && i < normalizedLassos.size(); ++i){
        if(normalizedLassos[i].attr1 != normalizedLassos[i].attr2 || 
            normalizedLassos[i].borderPoints.size() != _curLassoBrushes[i].borderPoints.size()){
            change = true;
            break;
        }
        for(int j = 0; !change && j < normalizedLassos[i].borderPoints.size(); ++j){
            if(normalizedLassos[i].borderPoints[j].x != _curLassoBrushes[i].borderPoints[j].x ||
                normalizedLassos[i].borderPoints[j].y != _curLassoBrushes[i].borderPoints[j].y)
                change = true;
        }
    }
    if(!change)     //don't do any updates if nothing has changed
        return;

    std::cout << "HierarchyImportManager::notifyBrushUpdate(): brushChanged..." << std::endl;

    _curRangeBrushes = normalizedBrushes;
    std::cout << "RangeBrushes:" << std::endl;
    for(auto& b: _curRangeBrushes){
        std::cout << "RangeBrush:" << std::endl;
        for(auto& r: b){
            std::cout<< "    " << r.axis << ":" << r.min << "|" << r.max << std::endl;
        }
    }
    _curLassoBrushes = normalizedLassos;

    //after normalization go through the hierarchy to check the level sizes beginning at the top most level
    std::vector<std::string_view> bestFiles;
    std::vector<std::vector<size_t>> bestFileOffsets;
    switch(_cachingMethod){
    case compression::CachingMethod::Bundled: {
        for(int i = _baseLevel; i < _levelFiles.size(); ++i){
            std::vector<std::string_view> curFiles{};
            std::vector<size_t> curFileOffsets{};
            size_t curLineCount{};
            for(auto& f: _levelInfos[i]){
                std::ifstream in(std::string(f), std::ios_base::binary);
                //getting the header informations
                uint32_t colCount, dataOffset, byteSize, symbolsSize, dataSize, o;
                float quantizationStep, eps;
                o = in.tellg();
                while(in >> colCount >> dataOffset >> byteSize >> symbolsSize >> dataSize >> quantizationStep >> eps){
                    std::vector<float> center(colCount);
                    for(int i = 0; i < colCount; ++i)
                        in >> center[i];
                    if(inBrush(_curRangeBrushes, _curLassoBrushes, center, eps)){
                        curLineCount += dataSize / colCount;
                        if(curFiles.empty())
                            curFiles.push_back(f);
                        curFileOffsets.push_back(size_t(o));
                    }
                    if(curLineCount > _maxLines){
                        std::cout << "HierarchyImportManager::notifyBrushUpdate(): _maxLines reached at level " << i << " with " << curLineCount << "lines" << std::endl;
                        goto doneFinding;           //exits both loops
                    }
                    o = in.tellg();
                } 
            }
            bestFiles = curFiles;
            bestFileOffsets.clear();
            bestFileOffsets.push_back(curFileOffsets);
        }
        break;
    }
    default:
        for(int i = _baseLevel; i < _levelFiles.size(); ++i){
            std::vector<std::string_view> curFiles;
            size_t curLineCount{};
            for(auto& f: _levelFiles[i]){
                //getting the header informations
                std::ifstream in(std::string(f), std::ios_base::binary);
                uint32_t colCount, byteSize, symbolsSize, dataSize;
                float quantizationStep, eps;
                in >> colCount >> byteSize >> symbolsSize >> dataSize >> quantizationStep >> eps;
                std::vector<float> center(colCount);
                for(int i = 0; i < colCount; ++i)
                    in >> center[i];
                in.close();
                if(inBrush(rangeBrushes, lassoBrushes, center, eps)){
                    curLineCount += dataSize / colCount;
                    curFiles.push_back(f);
                }
                if(curLineCount > _maxLines)
                    goto doneFinding;           //exits both loops
            }
            bestFiles = curFiles;
        } 
        break;
    }
    doneFinding:
    openHierarchyFiles(bestFiles, bestFileOffsets);          //opens the new hierarchy level
}

void HierarchyImportManager::checkPendingFiles() 
{
    using namespace std::chrono_literals;
    if(_dataLoadFuture.valid() && _dataLoadFuture.wait_for(0s) != std::future_status::timeout && _enqueuedFiles.size() != 0)
        openHierarchyFiles(_enqueuedFiles, _enqueuedBundles);
}

void HierarchyImportManager::openHierarchyFiles(const std::vector<std::string_view>& files, const std::vector<std::vector<size_t>> bundleOffsets){
    using namespace std::chrono_literals;
    // if loading is still active return and dont open new hierarchy files, but cache new files
    if(_dataLoadFuture.valid() && _dataLoadFuture.wait_for(0s) == std::future_status::timeout) {
        _enqueuedFiles = files;
        _enqueuedBundles = bundleOffsets;
        return;
    }
    
    auto exec = [](std::vector<std::string_view> files, HierarchyImportManager* m){
        std::vector<Data> dataVec(files.size());
        uint32_t i = 0;
        for(auto& f: files){
            compression::loadAndDecompress(f, dataVec[i++]);
        }
        compression::combineData(dataVec, m->_nextData);
        //denormalizing from [0,1] to [min,max]
        for(int a = 0; a < m->_attributes.size(); ++a){
            float diff = m->_attributes[a].max - m->_attributes[a].min;
            for(i = 0; i < m->_nextData.columns[a].size(); ++i)
                m->_nextData.columns[a][i] = m->_nextData.columns[a][i] * diff + m->_attributes[a].min;
        }
        std::cout << "HierarchyImportManager::openHierarchyFiles() loaded new data with " << m->_nextData.size() << " datapoints" << std::endl;
        m->newDataLoaded = true;
    };

    auto execBundled = [](std::vector<std::string_view> infos, std::vector<std::vector<size_t>> bundleOffsets, HierarchyImportManager* m){
        if(infos.empty())
            return;
        uint32_t dataBlocks{}; for(const auto& e: bundleOffsets) dataBlocks += e.size();
        std::vector<Data> dataVec(dataBlocks);
        uint32_t dataIndex{};
        assert(infos.size() == bundleOffsets.size());
        for(int i = 0; i < infos.size(); ++i){
            for(int j = 0; j < bundleOffsets[i].size(); ++j){
                compression::loadAndDecompressBundled(infos[i], bundleOffsets[i][j], dataVec[dataIndex++]);
            }
        }

        compression::combineData(dataVec, m->_nextData);
        //denormalizing from [0,1] to [min,max]
        for(int a = 0; a < m->_attributes.size(); ++a){
            float diff = m->_attributes[a].max - m->_attributes[a].min;
            for(int i = 0; i < m->_nextData.columns[a].size(); ++i)
                m->_nextData.columns[a][i] = m->_nextData.columns[a][i] * diff + m->_attributes[a].min;
        }
        std::cout << "HierarchyImportManager::openHierarchyFiles() loaded new data with " << m->_nextData.size() << " datapoints" << std::endl;
        m->newDataLoaded = true;
    };

    switch(_cachingMethod){
    case compression::CachingMethod::Bundled:
        _dataLoadFuture = std::async(execBundled, files, bundleOffsets, this);
        break;
    default:
        _dataLoadFuture = std::async(exec, files, this);
        break;
    } 
}
