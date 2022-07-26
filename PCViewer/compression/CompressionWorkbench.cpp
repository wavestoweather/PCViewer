#include "CompressionWorkbench.hpp"
#include "../imgui/imgui_stdlib.h"
#include "NetCdfLoader.hpp"
#include "../PCUtil.h"
#include "HirarchyCreation.hpp"
#include <thread>
#include <chrono>
#include <algorithm>
#include "Constants.hpp"

void analyse(std::shared_ptr<DataLoader> loader, size_t* dataSize, std::vector<Attribute>* attributes){
    loader->dataAnalysis(*dataSize, *attributes);
}

void CompressionWorkbench::draw() 
{
    if(!active) return;

    if(ImGui::Begin("CompresssionWorkbench", &active)){
        ImGui::Text("Open data files:");
        ImGui::InputText("Src Directory/File", &_inputFiles);
        ImGui::BeginChild("includes", {ImGui::GetWindowWidth() / 2.2f, 200});
        if(ImGui::CollapsingHeader("Include Formats"))
        {
            for(int i = 0; i < _includedFiles.size(); ++i){
                ImGui::InputText(("##inc" + std::to_string(i)).c_str(), &_includedFiles[i]);
            }
            if(_includedFiles.size() && ImGui::Button("Remove last Format")){
                _includedFiles.pop_back();
            }
            if(ImGui::Button("Add Include Format")){
                _includedFiles.push_back({});
            }
        }
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("excludes", {ImGui::GetWindowWidth() / 2.2f, 200});
        if(ImGui::CollapsingHeader("Exclude Formats"))
        {
            for(int i = 0; i < _excludedFiles.size(); ++i){
                ImGui::InputText(("##exc" + std::to_string(i)).c_str(), &_excludedFiles[i]);
            }
            if(_excludedFiles.size() && ImGui::Button("Remove last Format")){
                _excludedFiles.pop_back();
            }
            if(ImGui::Button("Add Exclude Format")){
                _excludedFiles.push_back({});
            }
        }
        ImGui::EndChild();
        //if(ImGui::Button("Create data loader")){
        //    std::vector<std::string_view> includeView(_includedFiles.begin(), _includedFiles.end());
        //    std::vector<std::string_view> excludeView(_excludedFiles.begin(), _excludedFiles.end());
        //    try{
        //        _loader = std::make_shared<NetCdfLoader>(_inputFiles, includeView, excludeView);
        //    }
        //    catch(std::runtime_error e){
        //        std::cout << e.what() << std::endl;
        //    }
        //}
        if(ImGui::Button("Create Column loader")){
            std::vector<std::string_view> includeView(_includedFiles.begin(), _includedFiles.end());
            std::vector<std::string_view> excludeView(_excludedFiles.begin(), _excludedFiles.end());
            try{
                _columnLoader = std::make_shared<NetCdfColumnLoader>(_inputFiles, includeView, excludeView);
            }
            catch(std::runtime_error e){
                std::cout << e.what() << std::endl;
            }
        }
        if(_loader){
            ImGui::Text("Loader contains %i files", static_cast<int>(_loader->getFileAmt()));
            if(ImGui::CollapsingHeader("Data dimension settings")){
                for(auto& e: _loader->queryAttributes){
                    if(e.dimensionSize == 0) 
                        ImGui::Checkbox(e.name.c_str(), &e.active);
                }
            }
        }

        if(_columnLoader){
            ImGui::Text("Column loader contains %i files", static_cast<int>(_columnLoader->getFileAmt()));
            if(ImGui::CollapsingHeader("Data dimension settings")){
                for(auto& a: _columnLoader->queryAttributes){
                    if (a.dimensionSize > 0) {
                        ImGui::Text("%s", a.name.c_str());
						ImGui::PushItemWidth(100);
						if (a.active) {
							ImGui::SameLine();
							ImGui::InputInt(("sampleFrequency##" + a.name).c_str(), &a.dimensionSubsample);
							if (a.dimensionSubsample < 1) a.dimensionSubsample = 1;
							ImGui::SameLine();
							ImGui::InputInt2(("Trim Indices##" + a.name).c_str(), a.trimIndices);
						}
						else {
							ImGui::SameLine();
							ImGui::InputInt(("slice Index##" + a.name).c_str(), &a.dimensionSlice);
							a.dimensionSlice = std::clamp(a.dimensionSlice, 0, a.dimensionSize - 1);
						}
						ImGui::PopItemWidth();
					}
                }
            }
        }
        if(_loader || _columnLoader && ImGui::Button("Analyze")){
            if(_loader)
                _analysisFuture = std::async(analyse, _loader, &_dataSize, &_attributes);
            else
                _columnDataAnalysisFuture = std::async(&ColumnLoader::dataAnalysis, _columnLoader.get());
            //_loader->dataAnalysis(_dataSize, _attributes);
        }
        using namespace std::chrono_literals;
        if(_columnDataAnalysisFuture.valid() && _columnDataAnalysisFuture.wait_for(0s) == std::future_status::ready){
            auto res = _columnDataAnalysisFuture.get();
            _dataSize = res.size;
            _attributes = res.attributes;
        }
        ImGui::Text("Analyzed data size: %u", static_cast<uint32_t>(_dataSize));
        if(_loader)
            ImGui::Text("Loader progress: %.2f%%", _loader->progress() * 100);
        if(_columnLoader)
            ImGui::Text("Loader progress: %.2f%%", _columnLoader->progress() * 100);
        
        ImGui::Separator();
        ImGui::InputText("Output path", &_outputFolder);
        //if(ImGui::InputFloat("Epsilon start", &_epsStart, .01f, .1f)) _epsStart = std::clamp(_epsStart, 1e-6f, 1.0f);
        //if(ImGui::InputInt("Lines per level", reinterpret_cast<int*>(&_linesPerLvl))) _linesPerLvl = std::clamp<uint32_t>(_linesPerLvl, 1, 1e7);
        //if(ImGui::InputInt("Levels", reinterpret_cast<int*>(&_levels))) _levels = std::clamp<uint32_t>(_levels, 1, 20);
        //if(ImGui::InputInt("Max ram usage in MBytes", reinterpret_cast<int*>(&_maxWorkingMemory))) _maxWorkingMemory = std::max(10u, _maxWorkingMemory);
        //if(ImGui::InputInt("Amt of threads", reinterpret_cast<int*>(&_amtOfThreads))) _amtOfThreads = std::max(1u, _amtOfThreads);
        //if(_analysisFuture.valid() && _analysisFuture.wait_for(0s) == std::future_status::ready && ImGui::Button("Build Hierarchy")){
        //    _buildHierarchyFuture = std::async(compression::createHirarchy, std::string_view(_outputFolder), _loader.get(), _epsStart, _levels, _linesPerLvl, _maxWorkingMemory, _amtOfThreads, _quantizationStep);
        //}
        //if(_buildHierarchyFuture.valid()){
        //    if(_loader)
        //        ImGui::Text("Hierarchy creation at %.2f%%", _loader->progress() * 100);
        //    else
        //        ImGui::Text("Hierarchy creation at %.2f%%", _columnLoader->progress() * 100);
        //}
        //ImGui::Separator();
        //if(ImGui::Button("Compress")){
        //    compression::compressTempHirarchy(_outputFolder, _amtOfThreads, _quantizationStep);
        //}
        //if(ImGui::Button("Convert")){
        //    compression::convertTempHierarchy(_outputFolder, _amtOfThreads);
        //}
        //ImGui::Separator();
        //if(ImGui::InputInt("Start Cluster Amount", reinterpret_cast<int*>(&_startCluster))) _startCluster = std::clamp<uint32_t>(_startCluster, 1, 20);
        //if(ImGui::InputInt("Cluster Multiplier", reinterpret_cast<int*>(&_clusterMultiplicator))) _clusterMultiplicator = std::clamp<uint32_t>(_clusterMultiplicator, 1, 20);
        //if(ImGui::InputInt("Cluster Dimensionality", reinterpret_cast<int*>(&_dimensionality))) _dimensionality = std::clamp<uint32_t>(_dimensionality, 1, 50);
        //if(ImGui::Button("Cluster ND hierarchy")){
        //    _buildHierarchyFuture = std::async(compression::createNDHierarchy, _outputFolder, _loader.get(), compression::CachingMethod::Bundled, _startCluster, _clusterMultiplicator, _dimensionality, _levels, _maxWorkingMemory, _amtOfThreads);
        //}
        //ImGui::Separator();
        if(ImGui::InputInt("1d cluster bin amount", reinterpret_cast<int*>(&_binsAmt))) _binsAmt = std::clamp<uint32_t>(_startCluster, 1, 1e6);
        if(ImGui::InputScalar("Compression block size(powerOf2)->", ImGuiDataType_U32, &_compressionBlockSize)) _compressionBlockSize = std::clamp(_compressionBlockSize, 1u, 31u);
        ImGui::SameLine(); ImGui::Text("%u", 1u << _compressionBlockSize);
        if(ImGui::InputFloat("Quantizatio step", &_quantizationStep, .0f,.0f, "%.4f")) _quantizationStep = std::clamp(_quantizationStep, .0001f, 1.f);
        ImGui::Checkbox("Float column data", &_floatColumnData);
        ImGui::Checkbox("Half column data", &_halfClolumnData);
        ImGui::Checkbox("Compressed column data", &_compressedColumnData);
        ImGui::Checkbox("Roaring column bin indices", &_roaringBinIndices);
        //if(ImGui::Button("Cluster 1d indices")){
        //    _buildHierarchyFuture = std::async(compression::create1DBinIndex, _outputFolder, _columnLoader.get(), _binsAmt, _maxWorkingMemory, _amtOfThreads);
        //}
        using namespace compression;
        if(ImGui::Button("Cluster 1d with compressed columns")){
            DataStorageBits storageBits{};
            if(_floatColumnData) storageBits |= DataStorageBits::RawColumnData;
            if(_halfClolumnData) storageBits |= DataStorageBits::HalfColumnData;
            if(_compressedColumnData) storageBits |= DataStorageBits::CuComColumnData;
            if(_roaringBinIndices) storageBits |= DataStorageBits::RoaringAttributeBins;
            _buildHierarchyFuture = std::async(compression::createRoaringBinsColumnData, _outputFolder, _columnLoader.get(), _binsAmt, static_cast<size_t>(1 << _compressionBlockSize), _quantizationStep, storageBits, _amtOfThreads);
        }
    }
    ImGui::End();
}

void CompressionWorkbench::stopThreads() 
{
    //canelling loading to also stop hirarchy building
    using namespace std::chrono_literals;
    if(_analysisFuture.valid() && _analysisFuture.wait_for(0s) == std::future_status::ready)
        _loader->reset();
}

CompressionWorkbench::~CompressionWorkbench() 
{

}