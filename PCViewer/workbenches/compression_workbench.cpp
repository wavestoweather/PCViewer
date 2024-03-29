#include "compression_workbench.hpp"
#include <logger.hpp>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <filesystem>
#include <regex>
#include <imgui_internal.h>
#include <dataset_util.hpp>
#include <fstream>
#include <json_util.hpp>
#include <util.hpp>
#include <as_cast.hpp>

static std::vector<std::string> get_data_filenames(const std::string_view& path, util::memory_view<const std::string> includes, util::memory_view<const std::string> ignores){
    // searching all files in the given directory (also in the subdirectories) and append all found netCdf files to the _files variable
    // all files and folders given in ignores will be skipped
    // if path is a netCdf file, only add the netcdf file
    std::vector<std::string> files;
    if(path.find_last_of(".") <= path.size() && path.substr(path.find_last_of(".")) == ".nc"){
        files.push_back(std::string(path));
    }
    else{
        auto isIgnored = [&](const std::string_view& n, util::memory_view<const std::string> ignores){
            return std::find_if(ignores.begin(), ignores.end(), [&](const std::string& s){return std::regex_search(n.begin(), n.end(), std::regex(s.data()));}) != ignores.end();
        };
        auto isIncluded = [&](const std::string_view& n, util::memory_view<const std::string> includes){
            if(includes.empty()) return true;
            return std::find_if(includes.begin(), includes.end(), [&](const std::string& s){return std::regex_search(n.begin(), n.end(), std::regex(s.data()));}) != includes.end();
        };
        std::vector<std::string> folders{std::string(path)};
        while(!folders.empty()){
            std::string curFolder = folders.back(); folders.pop_back();
            std::string_view folderName = std::string_view(curFolder).substr(curFolder.find_last_of("/\\"));
            if(!isIgnored(folderName, ignores)){
                // folder should not be ignored
                // get all contents and iterate over them
                for(const auto& entry: std::filesystem::directory_iterator(curFolder)){
                    if(entry.is_directory()){
                        folders.push_back(entry.path().string());
                    }
                    else if(entry.is_regular_file()){
                        // check if should be ignored
                        std::string filename = entry.path().filename().string();
                        if(isIncluded(filename, includes) && !isIgnored(filename, ignores) && filename.substr(filename.find_last_of(".")) == ".nc"){
                            files.push_back(entry.path().string());
                        }
                    }
                }
            }
        }
    }
    return files;
}

namespace workbenches
{
void compression_workbench::_analyse(std::vector<std::string> files, file_version version){
    if(_analysis_cancel){
        _analysis_cancel = false;
        return;
    }

    _analysis_running = true;

    analysed_data_t analysed_data{};
    try{
        int fc{};
        for(const auto& file: files){
            if(_analysis_cancel){
                _analysis_cancel = false;
                _analysis_running = false;
                return;
            }
            util::dataset::open_internals::load_result<float> load_result; 
            auto [filename, extension] = util::get_file_extension(file);
            if(extension == ".csv")
                load_result = util::dataset::open_internals::open_csv<float>(std::string_view(file));
            else if(extension == ".nc")
                load_result = util::dataset::open_internals::open_netcdf<float>(std::string_view(file));

            _analysis_progress = float(fc++) / files.size() / 2;

            // attribute consistency check
            if(analysed_data.attributes.empty())
                analysed_data.attributes = load_result.attributes;
            else{
                for(size_t a: util::size_range(load_result.attributes)){
                    if(analysed_data.attributes.size() - 1 < a || analysed_data.attributes[a].id != load_result.attributes[a].id){
                        _analysis_running = false;
                        throw std::runtime_error{"compression_workbench::_analyse() Attributes inconsistent for file " + file};
                    }
                }
            }

            // analyzing min/max
            for(size_t a: util::size_range(load_result.data.columns)){
                for(float f: load_result.data.columns[a]){
                    if(_analysis_cancel){
                        _analysis_cancel = false;
                        _analysis_running = false;
                        return;
                    }
                    
                    if(analysed_data.attributes[a].bounds.read().min > f)
                        analysed_data.attributes[a].bounds().min = f;
                    if(analysed_data.attributes[a].bounds.read().max < f)
                        analysed_data.attributes[a].bounds().max = f;
                }
            }

            analysed_data.data_size += load_result.data.size();

            _analysis_progress = float(fc++) / files.size() / 2;
        }
    }
    catch(std::exception e){
        if(logger.logging_level >= logging::level::l_3)
            logger << logging::error_prefix << " " << e.what() << logging::endl;
        _analysis_running = false;
        return;
    }
    

    analysed_data.files_version = version;
    _analysed_data.access() = std::move(analysed_data);
    _analysis_progress = 1;
    _analysis_running = false;
}

void compression_workbench::_compress(std::vector<std::string> files, std::string output_folder, analysed_data_t analysed_data, compress_info compress_info){
    if(output_folder.empty()){
        logger << logging::error_prefix << " compression_workbench::_compress() output folder empty" << logging::endl;
        return;
    }

    if(_compression_cancel){
        _compression_cancel = false;
        return;
    }

    _compression_running = true;

    std::vector<std::vector<float>> column_data(analysed_data.attributes.size());

    auto append_to_column_file = [&](int col){
        if(logger.logging_level >= logging::level::l_5)
            logger << logging::info_prefix << " compression_workbench::_compress() Append converted data block" << logging::endl;
        if(compress_info.half_column_data){
            std::string output_file = std::string(output_folder) + "/" + std::to_string(col) + ".col";
            structures::c_file column_file(std::string_view(output_file), "ab");
            assert(column_file);
            std::vector<half> write(column_data[col].begin(), column_data[col].end());
            column_file.write<half>(util::memory_view(write));
        }
        if(compress_info.compressed_column_data){
            // TODO compression reimplement
        }
        column_data[col].clear();
    };

    size_t offset{};
    for(const auto& file: files){
        // loading file
        auto data = util::dataset::open_data<float>(file);

        for(size_t a: util::size_range(data.columns)){
            for(int64_t i: util::i_range(data.size())){
                if(_compression_cancel){
                    _compression_cancel = false;
                    _compression_running = false;
                    return;
                }

                float axis_val = data(as<uint32_t>(i), as<uint32_t>(a));
                axis_val = util::normalize_val_for_range(axis_val, analysed_data.attributes[a].bounds.read().min, analysed_data.attributes[a].bounds.read().max);

                if(compress_info.half_column_data || compress_info.compressed_column_data){
                    column_data[a].push_back(axis_val);
                    if(column_data[a].size() >= compress_info.comression_block_size_shift)
                        append_to_column_file(as<int>(a));
                }
                // TODO other compression stuff
            }
        }

        offset += data.size();
        _compression_progress = float(offset) / analysed_data.data_size;

        if(logger.logging_level >= logging::level::l_5)
            logger << logging::info_prefix << " compression_workbench::_compress() File " << file << " progressed" << logging::endl;
    }

    // ----------------------------------------------------------------------------------------------
    // writeout of the data info (json info)
    // ----------------------------------------------------------------------------------------------
    crude_json::value info;
    info["data_size"] = double(analysed_data.data_size);
    info["half_data"] = compress_info.half_column_data;
    info["compressed_data"] = compress_info.compressed_column_data;
    // adding the copmression infos
    if(compress_info.compressed_column_data){
        std::string c{"compression_infos"};
        info[c]["quant_step"] = compress_info.quantization_step;
        info[c]["block_size"] = double(compress_info.comression_block_size_shift);
    }

    for(size_t a: util::size_range(analysed_data.attributes)){
        info["attributes"][a]["name"] = analysed_data.attributes[a].id;
        info["attributes"][a]["min"] = analysed_data.attributes[a].bounds.read().min;
        info["attributes"][a]["max"] = analysed_data.attributes[a].bounds.read().max;
    }

    util::json::save_json(std::string(output_folder) + "/info.json", info);

    _compression_progress = 1;
    _compression_running = false;
}

compression_workbench::compression_workbench(std::string_view id):
    workbench(id)
{
}

compression_workbench::~compression_workbench(){
    // cancelling the worker threads if they are doing work
    if(_analysis_thread.joinable()){
        if(logger.logging_level >= logging::level::l_4)
            logger << logging::warning_prefix << " ~compression_workbench() Interrupting analysis thread" << logging::endl;
        _analysis_cancel = true;
        _analysis_thread.join();
    }
    if(_compression_thread.joinable()){
        if(logger.logging_level >= logging::level::l_4)
            logger << logging::warning_prefix << " ~compression_workbench() Interrupting compression thread. This might lead to corrupt intermediate file." << logging::endl;
        _compression_cancel = true;
        _compression_thread.join();
    }
}

void compression_workbench::show() 
{
    if(!active)
        return;
    
    ImGui::Begin(id.data(), &active);
    ImGui::Text("Open data files:");
    bool input_file_change = false;
    input_file_change |= ImGui::InputText("Src Directory/File", &_input_files);
    ImGui::BeginChild("includes", {ImGui::GetWindowWidth() / 2.2f, 200});
    if(ImGui::CollapsingHeader("Include Formats"))
    {
        for(int i = 0; i < _included_files.size(); ++i){
            input_file_change |= ImGui::InputText(("##inc" + std::to_string(i)).c_str(), &_included_files[i]);
        }
        if(_included_files.size() && ImGui::Button("Remove last Format")){
            _included_files.pop_back();
            input_file_change = true;
        }
        if(ImGui::Button("Add Include Format")){
            _included_files.push_back({});
            input_file_change = true;
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("excludes", {ImGui::GetWindowWidth() / 2.2f, 200});
    if(ImGui::CollapsingHeader("Exclude Formats"))
    {
        for(int i = 0; i < _excluded_files.size(); ++i){
            input_file_change |= ImGui::InputText(("##exc" + std::to_string(i)).c_str(), &_excluded_files[i]);
        }
        if(_excluded_files.size() && ImGui::Button("Remove last Format")){
            _excluded_files.pop_back();
            input_file_change = true;
        }
        if(ImGui::Button("Add Exclude Format")){
            _excluded_files.push_back({});
            input_file_change = true;
        }
    }
    ImGui::EndChild();

    // updating the loaded files 
    if(input_file_change){
        ++_input_files_version;
        try{
            _current_files = get_data_filenames(_input_files, _included_files, _excluded_files);
            _current_files_active.resize(_current_files.size(), 1);
        }
        catch(std::exception e){
            if(logger.logging_level >= logging::level::l_4)
                logger << logging::info_prefix << " compression_workbench: File query failed (incomplete/imparsable path)" << logging::endl;
        }
    }

    if(ImGui::CollapsingHeader(("Available files(" + std::to_string(_current_files.size()) + ")").c_str())){
        for(size_t i: util::size_range(_current_files))
            ImGui::Checkbox(_current_files[i].c_str(), reinterpret_cast<bool*>(_current_files_active.data() + i));
    }

    // start/end analysis
    ImGui::Separator();
    ImGui::Text("Data analysis");
    ImGui::ProgressBar(_analysis_progress);
    bool disable_analysis_button = _analysis_running;
    ImGui::BeginDisabled(disable_analysis_button);
    if(ImGui::Button("Analyse")){
        std::vector<std::string> files;
        for(size_t i: util::size_range(_current_files)){
            if(_current_files_active[i])
                files.push_back(_current_files[i]);
        }
        _analysis_progress = 0;
        if(_analysis_thread.joinable())
            _analysis_thread.join();
        _analysis_thread = std::thread(&compression_workbench::_analyse, this, files, _input_files_version);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    bool disable_analysis_cancel_button = !_analysis_running;
    ImGui::BeginDisabled(disable_analysis_cancel_button);
    if(ImGui::Button("Cancel Analysis")){
        _analysis_cancel = true;
        _analysis_progress = 0;
        _analysis_thread.join();
        _analysis_thread = {};
    }
    ImGui::EndDisabled();

    ImGui::Separator();

    ImGui::Text("Compression");
    ImGui::InputText("Output folder", &_output_folder);
    ImGui::Checkbox("Float column data", &_compress_info.float_column_data);
    ImGui::Checkbox("Half column data", &_compress_info.half_column_data);
    ImGui::Checkbox("Compressed column data", &_compress_info.compressed_column_data);

    if(_analysed_data.const_access()->files_version != _input_files_version)
        ImGui::TextColored({1.f, .1f, .1f, 1.f}, "Filelist changed, first run analysis.");
    ImGui::ProgressBar(_compression_progress);
    ImGui::SameLine(); ImGui::Text("progress");
    bool disable_compression_button = _compression_running || _analysis_progress != 1 || _analysed_data.const_access()->files_version != _input_files_version;
    ImGui::BeginDisabled(disable_compression_button);
    if(ImGui::Button("Compress")){
        std::vector<std::string> files;
        for(size_t i: util::size_range(_current_files)){
            if(_current_files_active[i])
                files.push_back(_current_files[i]);
        }
        _compression_cancel = false;
        _compression_progress = 0;
        if(_compression_thread.joinable())
            _compression_thread.join();
        _compression_thread = std::thread(&compression_workbench::_compress, this, files, _output_folder, *_analysed_data.const_access(), _compress_info);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    bool disable_cancel_compression_button = !_compression_running;
    ImGui::BeginDisabled(disable_cancel_compression_button);
    if(ImGui::Button("Cancel Compress")){
        _compression_cancel = true;
        _compression_progress = 0;
        _compression_thread.join();
        _compression_thread = {};
    }
    ImGui::EndDisabled();

    ImGui::End();
}
}