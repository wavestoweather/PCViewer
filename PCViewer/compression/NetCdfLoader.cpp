#include "NetCdfLoader.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include "../PCUtil.h"
#include "../range.hpp"
#include <numeric>

static std::vector<std::string> getDataFilenames(const std::string_view& path, const std::vector<std::string_view>& includes, const std::vector<std::string_view>& ignores){
    // searching all files in the given directory (also in the subdirectories) and append all found netCdf files to the _files variable
    // all files and folders given in ignores will be skipped
    // if path is a netCdf file, only add the netcdf file
    std::vector<std::string> files;
    if(path.find_last_of(".") <= path.size() && path.substr(path.find_last_of(".")) == ".nc"){
        files.push_back(std::string(path));
    }
    else{
        auto isIgnored = [&](const std::string_view& n, const std::vector<std::string_view>& ignores){
            return std::find_if(ignores.begin(), ignores.end(), [&](const std::string_view& s){return PCUtil::compareStringFormat(n, s);}) != ignores.end();
        };
        auto isIncluded = [&](const std::string_view& n, const std::vector<std::string_view>& includes){
            if(includes.empty()) return true;
            return std::find_if(includes.begin(), includes.end(), [&](const std::string_view& s){return PCUtil::compareStringFormat(n, s);}) != includes.end();
        };
        std::vector<std::string> folders{std::string(path)};
        while(!folders.empty()){
            std::string curFolder = folders.back(); folders.pop_back();
            std::string_view folderName = std::string_view(curFolder).substr(curFolder.find_last_of("/\\"));
            if(!isIgnored(folderName, ignores)){
                // folder should not be ignored
                // get all contents and iterate over them
                for(const auto& entry: std::filesystem::directory_iterator(curFolder)){
                    if(entry.is_directory()){
                        folders.push_back(entry.path().string());
                    }
                    else if(entry.is_regular_file()){
                        // check if should be ignored
                        std::string filename = entry.path().filename().string();
                        if(isIncluded(filename, includes) && !isIgnored(filename, ignores) && filename.substr(filename.find_last_of(".")) == ".nc"){
                            files.push_back(entry.path().string());
                        }
                    }
                }
            }
        }
    }
    return files;
}

NetCdfLoader::NetCdfLoader(const std::string_view& path, const std::vector<std::string_view>& includes, const std::vector<std::string_view>& ignores)
{
    _files = getDataFilenames(path, includes, ignores);
    
    std::cout << "Found " << _files.size() << " netCdf files in the given path" << std::endl;

    if(_files.empty())
        throw std::runtime_error("NetCdfLoader::NetCdfLoader(...) Could not find any files.");
    
    queryAttributes = PCUtil::queryNetCDF(_files.front());
}

void NetCdfLoader::reset() 
{
    std::unique_lock<std::shared_mutex> lock(_readMutex);
    _curFile = {};
    _curData = {};
    _curDataIndex = {};
    _curTotalIndex = {};
}

void NetCdfLoader::dataAnalysis(size_t& dataSize, std::vector<Attribute>& attributes) 
{
    if(_attributes.size()){
        dataSize = _dataSize;
        attributes = _attributes;
        return;
    }
    std::cout << "Data analysis: 0%";
    std::cout.flush();
    _dataSize = 0;
    for(int i = 0; i < _files.size(); ++i){
        Data d = PCUtil::openNetCdf(_files[i], _attributes, queryAttributes);  //parses netcdf file and updates the _attributes vector
        _dataSize += d.size();
        _progress = (i + 1.0) / _files.size();
        std::cout << "\rData analysis: " << _progress * 100 << "%";
        std::cout.flush();
    }
    std::cout << std::endl;
    attributes = _attributes;
    dataSize = _dataSize;
}

bool NetCdfLoader::getNext(std::vector<float>& d) 
{
    std::unique_lock<std::shared_mutex> lock(_readMutex);   //making the method thread safe at reading, only letting a single thread read at a time
    if(_curData.size() == 0){
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, queryAttributes);
    }
    else if(_curData.size() == _curDataIndex){
        if(++_curFile >= _files.size()) 
            return false;
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, queryAttributes);
        _curDataIndex = 0;
    }
    _progress = ++_curTotalIndex / static_cast<float>(_dataSize);
    d.resize(_attributes.size());
    for(int i = 0; i < _attributes.size(); ++i){
        d[i] = _curData(_curDataIndex, i);
    }
    ++_curDataIndex;
    return true;
}

bool NetCdfLoader::getNextNormalized(std::vector<float>& d) 
{
    //getting normal data
    bool n = getNext(d);
    if(!n) return false;
    //normalizing
    for(int i = 0; i < _attributes.size(); ++i){
        d[i] = (d[i] - _attributes[i].min) / (_attributes[i].max - _attributes[i].min + NORM_EPS);
    }
    return true;
}

NetCdfColumnLoader::NetCdfColumnLoader(const std::string_view& path, const std::vector<std::string_view>& includes, const std::vector<std::string_view>& ignores) 
{
    _files = getDataFilenames(path, includes, ignores);
    
    std::cout << "Found " << _files.size() << " netCdf files in the given path" << std::endl;

    if(_files.empty())
        throw std::runtime_error("NetCdfLoader::NetCdfLoader(...) Could not find any files.");
    
    queryAttributes = PCUtil::queryNetCDF(_files.front());
}

NetCdfColumnLoader::DataInfo NetCdfColumnLoader::dataAnalysis(){
    if(_attributes.size()){
        return {_dataSize, _attributes};
    }
    std::cout << "Data analysis: 0%";
    std::cout.flush();
    _dataSize = 0;
    for(int i = 0; i < _files.size(); ++i){
        Data d = PCUtil::openNetCdf(_files[i], _attributes, queryAttributes);  //parses netcdf file and updates the _attributes vector
        _dataSize += d.size();
        _progress = (i + 1.0) / _files.size();
        std::cout << "\rData analysis: " << _progress * 100 << "%";
        std::cout.flush();
    }
    std::cout << std::endl;
    return {_dataSize, _attributes};
}

void NetCdfColumnLoader::normalize() 
{
    if(!_normalized){
        for(int c = 0; c < _curData.columns.size(); ++c){
            float diff = _attributes[c].max - _attributes[c].min;
            for(float& f: _curData.columns[c]){
                f = (f - _attributes[c].min) / diff;
            }
        }
    }
    _normalized = true;
}

void NetCdfColumnLoader::tabelize()
{
    if(!_tabelized){
        std::vector<uint32_t> increasing(_curData.dimensionSizes.size());
        std::iota(increasing.begin(), increasing.end(), 0); 
        for(int i: irange(_curData.columns)){
            if(_curData.columnDimensions[i] == increasing)
            //if(_curData.columnDimensions[i].size() == increasing.size())    //temporary test for compression
                continue;
            std::vector<float> newColumn(_curData.size());
            for(int j: irange(newColumn)){
                newColumn[j] = _curData(j, i);
            }
            _curData.columns[i] = newColumn;
        }
    }
    _tabelized = true;
}

bool NetCdfColumnLoader::loadNextData() 
{
    if(_curData.size() == 0 || ++_curFile < _files.size()){
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, queryAttributes);
        if(_normalized){    // normalize if normalization is set
            _normalized = false;
            normalize();
        }
        if(_tabelized){     // tabelize if tablization is set
            _tabelized = false;
            tabelize();
        }
        _progress = (_curFile + 1.0) / _files.size();
        return true;
    }
    else
        return false;
}
