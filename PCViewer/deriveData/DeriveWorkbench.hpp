#pragma once
#include "../WindowBases.hpp"
#include <vector>

namespace ax::NodeEditor{
    class EditorContext;
}
struct ExecutionGraph;


class DeriveWorkbench: public Workbench, public DatasetDependency{
public:
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
};