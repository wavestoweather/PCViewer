#include"BundlingCacheManager.hpp"
#include<filesystem>
#include<fstream>
BundlingCacheManager::BundlingCacheManager(const std::string_view& temp) :_outputFolder(temp)
{
    
}

void BundlingCacheManager::addData(const std::string_view& nodeId, const std::stringstream& data) 
{
    _headerInfo.push_back({std::string(nodeId), _dataStream.tellp(), data.str().size()});
    _dataStream << data.str();
}

void BundlingCacheManager::postDataInsert() 
{
    //getting the filenumber (Might change this to using a static variable for this)
    uint32_t fileNumber = 0;
    for(auto const& entry: std::filesystem::directory_iterator(_outputFolder)){
        if(entry.is_regular_file() && entry.path().filename().string().find("bundle"))
            ++fileNumber;
    }
    //opening the filestream and writing the data
    std::ofstream file(std::string(_outputFolder) + "/bundle" + std::to_string(fileNumber));
    file << _headerInfo.size() << "\n";
    size_t offset = file.tellp();
    for(const auto& i: _headerInfo){
        file << i.id << " " << static_cast<size_t>(i.start) + offset << " " << i.size << "\n";   // the offset is directly added to make the reading easier
    }
    file << _dataStream.str();
}

void BundlingCacheManager::finish(){
    
}
