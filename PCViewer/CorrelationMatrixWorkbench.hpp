#pragma once
#include "VkUtil.h"
#include "CorrelationManager.hpp"
#include <vector>
#include <list>
#include <string>
#include <memory>
#include "imgui/imgui.h"

class CorrelationMatrixWorkbench{
public:
    class CorrelationMatrix{
    public:
        struct DrawlistRef{
            std::string drawlist;
            bool active;
        };

        struct AttributeRef{
            std::string attribute;
            bool active;
        };

        CorrelationMatrix(const std::shared_ptr<CorrelationManager>& correlationManager);

        void draw();
        void addDrawlist(const DrawList& dl);
        // if a drawlist is not anymore int the dls array, that drawlist will be deleted
        void updateAllCorrelationScores(const std::list<DrawList>& dls);

        std::vector<DrawlistRef> drawlists;
        std::vector<AttributeRef> attributes;
        int matrixPixelWidth = 800;
        float matrixSpacing = .05;
        ImVec4 matrixBorderColor{1, 1, 1, 1};
        float matrixBorderWidth{1};
        CorrelationManager::CorrelationMetric currentMetric{CorrelationManager::CorrelationMetric::Pearson};
        bool requestCorrelationUpdate{};    // correlation updates have to be started from outside with the list of all drawlists
    private:
        static uint32_t idCounter;
        uint32_t id;
        std::shared_ptr<CorrelationManager> correlationManager;
    };

    CorrelationMatrixWorkbench(const VkUtil::Context& vkContext);

    void draw();    //draw Imgui window
    void updateCorrelationScores(const std::list<DrawList>& dls, bool force = false);   // force can be used to update drawlsits when a brush update came

    bool active{false};
    bool requestUpdate{false};
private:
    VkUtil::Context vkContext;
    std::shared_ptr<CorrelationManager> correlationManager;
    std::vector<CorrelationMatrix> correlationMatrices;
};
