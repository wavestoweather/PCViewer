#pragma once
#include <workbench_base.hpp>
#include <attributes.hpp>
#include <thread>
#include <atomic>
#include <sys_info.hpp>
#include <thread_safe_struct.hpp>

namespace workbenches{
class compression_workbench: public structures::workbench{
    typedef uint64_t file_version;
    struct compress_info{
        uint32_t comression_block_size_shift{28};   // power of 2
        uint32_t amt_of_threads{static_cast<uint32_t>(std::thread::hardware_concurrency() * .8)};
        uint32_t max_working_memory{static_cast<uint32_t>(globals::sys_info.ram_size * .5)};
        float    quantization_step{.01f};
        bool     float_column_data{false};
        bool     half_column_data{true};
        bool     compressed_column_data{false};
        bool     roaring_bin_indices{false};
    };

    std::string                         _input_files{};
    file_version                        _input_files_version{};
    std::vector<std::string>            _included_files{}, _excluded_files{};
    std::vector<std::string>            _current_files{};
    std::vector<uint8_t>                _current_files_active{};
    std::string                         _output_folder{};

    compress_info                       _compress_info{};

    // multithreading stuff for async loading/working
    std::thread                         _analysis_thread{};
    std::atomic<float>                  _analysis_progress{};
    std::atomic<bool>                   _analysis_running{};
    std::atomic<bool>                   _analysis_cancel{};
    std::thread                         _compression_thread{};
    std::atomic<float>                  _compression_progress{};
    std::atomic<bool>                   _compression_running{};
    std::atomic<bool>                   _compression_cancel{};

    struct analysed_data_t{
        file_version                        files_version{};
        size_t                              data_size{};
        std::vector<structures::attribute>  attributes{};
    };
    structures::thread_safe<analysed_data_t> _analysed_data{};

    void _analyse(std::vector<std::string> files, file_version version);            // analyses min/max + data size for all files listed in files. Takes a copy of the vector to avoid access problems
    void _compress(std::vector<std::string> files, std::string output_folder, analysed_data_t analysed_data, compress_info compress_info);
public:
    compression_workbench(std::string_view id);
    ~compression_workbench();

    void show() override;
};
}