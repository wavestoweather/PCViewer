#include "CsvLoader.hpp"
#include <iostream>

ColumnLoader::DataInfo CsvLoader::dataAnalysis() 
{
    if(_attributes.size())
        return {_data_size, _attributes};
    std::cout << "DAta analysis: 0%" << std::endl;
    _data_size = 0;

    return {};
}
