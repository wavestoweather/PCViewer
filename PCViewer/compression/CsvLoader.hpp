#pragma once

#include <string_view>
#include "DataLoader.hpp"
#include "../Data.hpp"
#include <mutex>
#include <shared_mutex>

class CsvLoader: public ColumnLoader{
    float _progress{0};
    int _cur_file_index{};
    std::vector<std::string> _files;
    bool _normalizes{false};
    Data _cur_data{};
    size_t _cur_data_index{};
    size_t _cur_total_index{};

    size_t _data_size{};
    std::vector<Attribute> _attributes{};

public:
    const float& progress() const override {return _progress;}
    DataInfo dataAnalysis() override;
    void normalize() override{};
    void tabelize() override{};
    Data& curData() override {return _cur_data;}
    bool loadNextData() override{return false;};
    size_t getFileAmt() override {return _files.size();}
};