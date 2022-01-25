#include "NetCdfLoader.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include "../PCUtil.h"

NetCdfLoader::NetCdfLoader(const std::string_view& path, const std::vector<std::string_view>& includes, const std::vector<std::string_view>& ignores)
{
    // searching all files in the given directory (also in the subdirectories) and append all found netCdf files to the _files variable
    // all files and folders given in ignores will be skipped
    // if path is a netCdf file, only add the netcdf file
    if(path.find_last_of(".") <= path.size() && path.substr(path.find_last_of(".")) == ".nc"){
        _files.push_back(std::string(path));
    }
    else{
        auto isIgnored = [&](const std::string_view& n, const std::vector<std::string_view>& ignores){
            return std::find_if(ignores.begin(), ignores.end(), [&](const std::string_view& s){return PCUtil::compareStringFormat(n, s);}) != ignores.end();
        };
        auto isIncluded = [&](const std::string_view& n, const std::vector<std::string_view>& includes){
            if(includes.empty()) return true;
            return std::find_if(includes.begin(), includes.end(), [&](const std::string_view& s){return PCUtil::compareStringFormat(n, s);}) != includes.end();
        };
        std::vector<std::string_view> folders{path};
        while(!folders.empty()){
            std::string_view curFolder = folders.back(); folders.pop_back();
            std::string_view folderName = curFolder.substr(curFolder.find_last_of("/\\"));
            if(!isIgnored(folderName, ignores)){
                // folder should not be ignored
                // get all contents and iterate over them
                for(const auto& entry: std::filesystem::directory_iterator(curFolder)){
                    if(entry.is_directory()){
                        folders.push_back(entry.path().string());
                    }
                    else if(entry.is_regular_file()){
                        // check if should be ignored
                        std::string_view filename = entry.path().string().substr(entry.path().string().find_last_of("/\\"));
                        if(isIncluded(filename, includes) && !isIgnored(filename, ignores) && filename.substr(filename.find_last_of(".")) == ".nc"){
                            _files.push_back(entry.path().string());
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "Found " << _files.size() << " netCdf files in the given path" << std::endl;

    if(_files.empty())
        throw std::runtime_error("NetCdfLoader::NetCdfLoader(...) Could not find any files.");
    
    queryAttributes = PCUtil::queryNetCDF(_files.front());
}

void NetCdfLoader::reset() 
{
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
        _progress = (i + 1.0) / _files.size();
        Data d = PCUtil::openNetCdf(_files[i], _attributes, queryAttributes);  //parses netcdf file and updates the _attributes vector
        _dataSize += d.size();
        std::cout << "\rData analysis: " << _progress * 100 << "%";
        std::cout.flush();
    }
    std::cout << std::endl;
    attributes = _attributes;
    dataSize = _dataSize;
}

bool NetCdfLoader::getNext(std::vector<float>& d) 
{
    if(_curData.size() == 0){
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, queryAttributes);
    }
    else if(_curData.size() == _curDataIndex){
        if(++_curFile == _files.size()) 
            return false;
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, queryAttributes);
        _curDataIndex = 0;
    }
    _progress = ++_curTotalIndex / static_cast<float>(_dataSize);
    d.resize(_attributes.size());
    for(int i = 0; i < _attributes.size(); ++i){
        d[i] = _curData(_curDataIndex, i);
    }
    return true;
}
