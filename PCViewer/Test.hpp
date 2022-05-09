#pragma once
#include "VkUtil.h"

// function which is called once at the beginning of the PCViewer which servers as indirection to speedup test compilations
struct TestInfo{    //additionally needed test info
    VkFramebuffer pcFramebuffer;
    VkRenderPass pcNoClearPass;
};
void TEST(const VkUtil::Context& context, const TestInfo& testInfo = {});