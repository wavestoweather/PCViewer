#pragma once

#include"CacheManagerInterface.hpp"
#include<vector>

class BundlingCacheManager: public CacheManagerInterface{
public:
    BundlingCacheManager(const std::string_view& temp);

    void addData(const std::string_view& nodeId, const std::stringstream& data);
    // for the bundlingCacheManager the data is written out each time when this function is called
    void postDataInsert();
    //does nothing
    void finish();
private:
    struct IdInfo{
        std::string id;
        std::streampos start;
        std::size_t size;
    };
    std::vector<IdInfo> _headerInfo;
    std::stringstream _dataStream;
    std::string _outputFolder;
};