#pragma once
#include "VkUtil.h"
#include "CorrelationManager.hpp"
#include <vector>
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

        CorrelationMatrix(std::shared_ptr<CorrelationManager>& CorrelationManager);

        void draw();
        void addDrawlist(const DrawList& dl);

        std::vector<DrawlistRef> drawlists;
        std::vector<AttributeRef> attributes;
        int matrixPixelWidth = 500;
        float matrixSpacing = .05;
        ImVec4 matrixBorderColor{1, 1, 1, 1};
        float matrixBorderWidth = 1;

    private:
        static uint32_t idCounter;
        uint32_t id;
        std::shared_ptr<CorrelationManager> correlationManager;
    };

    CorrelationMatrixWorkbench(const VkUtil::Context& vkContext);

    void draw();    //draw Imgui window

    bool active = false;
private:
    VkUtil::Context vkContext;
    std::shared_ptr<CorrelationManager> correlationManager;
    std::vector<CorrelationMatrix> correlationMatrices;
};
