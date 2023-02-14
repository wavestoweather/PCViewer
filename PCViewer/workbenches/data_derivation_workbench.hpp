#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <robin_hood.h>
#include <memory>
#include <enum_names.hpp>
#include <ExecutionGraph.hpp>
#include <data_derivation_structures.hpp>

namespace workbenches{
class data_derivation_workbench: public structures::workbench, public structures::dataset_dependency{
public:
    const std::string_view main_execution_graph_id{"main"};

    data_derivation_workbench(std::string_view id);
    ~data_derivation_workbench();

    void show() override;

    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override{};
    void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, structures::dataset_dependency::update_flags flags, const structures::gpu_sync_info& sync_info = {}) override{};
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override{};

private:
    using unique_execution_graph_map = std::map<std::string, std::unique_ptr<ExecutionGraph>>;
    using workbench_settings = structures::data_derivation::workbench_settings;
    ax::NodeEditor::EditorContext*  _editor_context{};
    unique_execution_graph_map      _execution_graphs{};    // each execution graph describes a function and can be called, main graph is called "main"

    int64_t                         _cur_id{1};
    int64_t                         _new_link_pin_id{};
    int64_t                         _context_node_id{};
    int64_t                         _context_pin_id{};
    int64_t                         _context_link_id{};
    ImVec2                          _popup_pos{};
    std::vector<std::string>        _cur_dimensions{};

    bool                            _create_new_node{false};

    workbench_settings              _settings;

    VkCommandPool                   _compute_command_pool{};
    VkFence                         _compute_fence{};

    bool _is_input_pin(int64_t pin_id);
    std::set<int64_t> _get_active_links_recursive(int64_t node);
    void _execute_graph(std::string_view id);

    struct recursion_data{
        struct node_info{
            deriveData::float_column_views  output_views;
            std::vector<int>                output_counts;
        };
        struct print_info{
            deriveData::Nodes::Node*       serialization_node;
            deriveData::float_column_views data;
        };
        std::set<int64_t>                       active_links{};
        std::vector<std::vector<float>>         data_storage{};
        std::map<int64_t, node_info>            node_infos{};
        std::vector<std::unique_ptr<uint32_t>>  create_vector_sizes{};
        std::stringstream                       op_codes_list{};
        std::vector<print_info>                 print_infos{};
    };
    void _build_cache_recursive(int64_t node, recursion_data& data);
};
}