#pragma once
#include "../WindowBases.hpp"
#include "ExecutionGraph.hpp"
#include <vector>
#include <array>

namespace ax::NodeEditor{
    class EditorContext;
}

class DeriveWorkbench: public Workbench, public DatasetDependency{
public:
    enum class Execution: uint32_t{
        Cpu,
        Gpu,
        COUNT
    };
    const std::array<std::string_view, static_cast<size_t>(Execution::COUNT)> ExecutionNames{
        "Cpu",
        "Gpu"
    };

    DeriveWorkbench();
    ~DeriveWorkbench();

    void show() override;
    void addDataset(std::string_view datasetId) override;
    void removeDataset(std::string_view datasetId) override;


private:
    ax::NodeEditor::EditorContext* _editorContext{};
    std::vector<ExecutionGraph> _executionGraphs{};

    float _leftPaneWidth{400};
    float _rightPaneWidth{800};
    long _curId{1};
    bool _createNewNode{false};
    ImVec2 _popupPos{};
    int _newLinkPinId{0};
    int _contextNodeId{};
    int _contextPinId{};
    int _contextLinkId{};

    bool isInputPin(long pinId);
    void executeGraph();
};