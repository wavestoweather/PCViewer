#pragma once

#include<string_view>
#include<vector>
#include<sstream>

class CacheManagerInterface{
public:
    virtual void addData(const std::string_view& nodeId, const std::stringstream& data) = 0;
    virtual void postDataInsert() = 0;      // method called after the caching data has been added to do some cleanup, first writouts etc.
    virtual void finish() = 0;              // method to finish the cache manager and do final writeouts
};