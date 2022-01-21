#pragma once

#include <string_view>
#include "DataLoader.hpp"
#include "../Data.hpp"

class NetCdfLoader: public DataLoader{
public:
    NetCdfLoader(const std::string_view& path, const std::vector<std::string_view>& ignores, const std::vector<QueryAttribute>& queriedAttributes);

    const float& progress() const {return _progress;};
    void dataAnalysis(size_t& dataSize, std::vector<Attribute>& attributes);
    bool getNext(std::vector<float>& d);
    void reset();
private:
    // file information
    int _curFile{};
    std::vector<std::string_view> _files{};
    Data _curData{};
    size_t _curDataIndex{};
    size_t _curTotalIndex{};

    // data analysis information
    size_t _dataSize{};   // amount of data points
    std::vector<QueryAttribute> _queryAttributes;
    std::vector<Attribute> _attributes;

    float _progress{};
};