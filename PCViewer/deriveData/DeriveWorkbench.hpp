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
    struct RecursionData{
        std::vector<std::vector<float>> dataStorage;
        struct NodeInfo{
            std::vector<deriveData::memory_view<float>> dataView;
            int waitCount;          // count to indicate how many parents have to be evaluated (inserted constants are ignored)
            int copyCount;          // count to indicate how often the output of the node has to be copied until consumed
        };
        std::map<long, NodeInfo> nodeInfos{};
    };
    void buildCacheRecursive(long node, RecursionData& data);
};