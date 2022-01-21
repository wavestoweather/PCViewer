#include "NetCdfLoader.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include "../PCUtil.h"

NetCdfLoader::NetCdfLoader(const std::string_view& path, const std::vector<std::string_view>& ignores,  const std::vector<QueryAttribute>& queriedAttributes):
_queryAttributes(queriedAttributes)
{
    // searching all files in the given directory (also in the subdirectories) and append all found netCdf files to the _files variable
    // all files and folders given in ignores will be skipped
    // if path is a netCdf file, only add the netcdf file
    if(path.substr(path.find_last_of(".")) == ".nc"){
        _files.push_back(path);
    }
    else{
        auto compareString = [](const std::string_view& s, const std::string_view& form){
            std::size_t curPos = 0, sPos = 0;
            while(curPos != std::string_view::npos){
                std::size_t nextPos = form.find("*", curPos);
                std::string_view curPart = form.substr(curPos, nextPos);
                if(s.find(curPart, sPos) == std::string_view::npos)
                    return false;
                sPos += curPart.size();
                curPos = nextPos + 1;
            }
            return true;
        };
        auto isIgnored = [compareString](const std::string_view& n, const std::vector<std::string_view>& ignores){
            return std::find_if(ignores.begin(), ignores.end(), [&](std::string_view& s){return compareString(n, s);}) != ignores.end();
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
                        folders.push_back(entry.path().c_str());
                    }
                    else if(entry.is_regular_file()){
                        // check if should be ignored
                        std::string_view filename = entry.path().string().substr(entry.path().string().find_last_of("/\\"));
                        if(!isIgnored(filename, ignores) && filename.substr(filename.find_last_of(".")) == "nc"){
                            _files.push_back(entry.path().c_str());
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "Found " << _files.size() << " netCdf files in the given path";

    if(_files.empty())
        throw std::runtime_error("NetCdfLoader::NetCdfLoader(...) Could not find any files.");
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
    std::cout << "Data analysis: 0%";
    _dataSize = 0;
    for(int i = 0; i < _files.size(); ++i){
        _progress = (i + 1.0) / _files.size();
        Data d = PCUtil::openNetCdf(_files[i], _attributes, _queryAttributes);  //parses netcdf file and updates the _attributes vector
        _dataSize += d.size();
        std::cout << "\rData analysis: " << _progress * 100 << "%";
        std::cout.flush();
    }
    attributes = _attributes;
    dataSize = _dataSize;
}

bool NetCdfLoader::getNext(std::vector<float>& d) 
{
    if(_curData.size() == 0){
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, _queryAttributes);
    }
    else if(_curData.size() == _curDataIndex){
        if(++_curFile == _files.size()) 
            return false;
        _curData = PCUtil::openNetCdf(_files[_curFile], _attributes, _queryAttributes);
        _curDataIndex = 0;
    }
    _progress = ++_curTotalIndex / static_cast<float>(_dataSize);
    d.resize(_attributes.size());
    for(int i = 0; i < _attributes.size(); ++i){
        d[i] = _curData(_curDataIndex, i);
    }
    return true;
}
