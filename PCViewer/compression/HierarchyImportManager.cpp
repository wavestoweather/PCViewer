#define NOSTATICS
#include "HierarchyImportManager.hpp"
#undef NOSTATICS
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "HirarchyCreation.hpp"

HierarchyImportManager::HierarchyImportManager(const std::string_view& hierarchyFolder, uint32_t maxDrawLines) :
_maxLines(maxDrawLines), _hierarchyFolder(hierarchyFolder)
{
    // loading all hierarchy files
    bool foundInfoFile{false};
    for(const auto& entry: std::filesystem::directory_iterator(hierarchyFolder)){
        if(entry.is_regular_file()){
            if(!entry.path().has_extension())   //standard compressed hierarchy file
                _hierarchyFiles.push_back(entry.path().string());
            else if(entry.path().extension().string() == ".info"){ //configuration file containing column information
                foundInfoFile = true;
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

    openHierarchyFiles(_levelFiles[_baseLevel]);
}

void HierarchyImportManager::notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes) 
{
    //converting the brushes to local coordinates (the normalized coordinates)
    std::vector<RangeBrush> normalizedBrushes;
    for(auto& b: rangeBrushes){
        RangeBrush rb{};
        for(auto& r: b){
            brushing::AxisRange ar{};
            ar.axis = r.axis;
            double diff = _attributes[r.axis].max - _attributes[r.axis].min;
            ar.min = (r.min - _attributes[r.axis].min) / diff;
            ar.max = (r.max - _attributes[r.axis].max) / diff;
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

    _curRangeBrushes = normalizedBrushes;
    _curLassoBrushes = normalizedLassos;

    //after normalization go through the hierarchy to check the level sizes beginning at the top most level
    std::vector<std::string_view> bestFiles;
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
    doneFinding:
    openHierarchyFiles(bestFiles);          //opens the new hierarchy level
}

void HierarchyImportManager::checkPendingFiles() 
{
    using namespace std::chrono_literals;
    if(_dataLoadFuture.valid() && _dataLoadFuture.wait_for(0s) != std::future_status::timeout && _enqueuedFiles.size() != 0)
        openHierarchyFiles(_enqueuedFiles);
}

void HierarchyImportManager::openHierarchyFiles(const std::vector<std::string_view>& files){
    using namespace std::chrono_literals;
    // if loading is still active return and dont open new hierarchy files, but cache new files
    if(_dataLoadFuture.valid() && _dataLoadFuture.wait_for(0s) == std::future_status::timeout) {
        _enqueuedFiles = files;
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

    _dataLoadFuture = std::async(exec, files, this);
}
