#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <robin_hood.h>
#include <memory>
#include <enum_names.hpp>
#include <ExecutionGraph.hpp>

namespace workbenches{
class data_derivation_workbench: public structures::workbench, public structures::dataset_dependency{
public:
    enum class execution: uint32_t{
        Cpu,
        Gpu,
        COUNT
    };
    structures::enum_names<execution> execution_names{
        "Cpu",
        "Gpu"
    };

    const std::string_view main_execution_graph_id{"main"};

    data_derivation_workbench(std::string_view id);

    void show() override;

    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override{};
    void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, structures::dataset_dependency::update_flags flags, const structures::gpu_sync_info& sync_info = {}) override{};
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override{};

private:
    using unique_execution_graph_map = std::map<std::string, std::unique_ptr<ExecutionGraph>>;
    ax::NodeEditor::EditorContext*  _editor_context{};
    unique_execution_graph_map      _execution_graphs{};    // each execution graph describes a function and can be called, main graph is called "main"

    int64_t                         _cur_id{};
    int                             _new_link_pin_id{};
    int                             _context_node_id{};
    int                             _context_pin_id{};
    int                             _conetxt_link_id{};

    bool is_input_pin(int64_t pin_id){return {};};
    std::set<int64_t> get_active_links_recursive(int64_t node){return {};};
    void execute_graph(std::string_view id){};

    struct recursion_data{
        struct node_info{
            deriveData::float_column_views  output_view;
            std::vector<int>                output_counts;
        };
        std::set<int64_t> active_links{};
        std::vector<std::vector<float>> data_storage{};
        std::map<int64_t, node_info> node_infos{};
        std::vector<std::unique_ptr<uint32_t>> create_vector_sizes; 
    };
    void build_cache_recursive(int64_t node, recursion_data& data);
};
}