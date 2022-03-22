#pragma once

#include <string_view>
#include "DataLoader.hpp"
#include "../Data.hpp"
#include <mutex>
#include <shared_mutex>

#define NORM_EPS  1e-8

class NetCdfLoader: public DataLoader{
public:
    NetCdfLoader(const std::string_view& path, const std::vector<std::string_view>& includes, const std::vector<std::string_view>& ignores);

    const float& progress() const {return _progress;};
    void dataAnalysis(size_t& dataSize, std::vector<Attribute>& attributes);
    bool getNext(std::vector<float>& d);
    bool getNextNormalized(std::vector<float>& d);
    void reset();
    size_t getFileAmt(){return _files.size();};
private:
    // file information
    int _curFile{};
    std::vector<std::string> _files{};
    Data _curData{};
    size_t _curDataIndex{};
    size_t _curTotalIndex{};

    // data analysis information
    size_t _dataSize{};   // amount of data points
    std::vector<Attribute> _attributes;

    float _progress{};

    std::shared_mutex _readMutex;
};

class NetCdfColumnLoader: public ColumnLoader{
public:
    NetCdfColumnLoader(const std::string_view& path, const std::vector<std::string_view>& includes, const std::vector<std::string_view>& ignores);

    const float& progress() const {return _progress;};
    DataInfo dataAnalysis();
    void normalize();
    Data& curData(){return _curData;};
    bool loadNextData();
    size_t getFileAmt(){return _files.size();};

private:
    float _progress{0};
    int _curFile{};
    std::vector<std::string> _files;
    bool _normalized{false};
    Data _curData{};
    size_t curDataIndex{};
    size_t _curTotalIndex{};

    // data analysis information
    size_t _dataSize{};
    std::vector<Attribute> _attributes{};
};