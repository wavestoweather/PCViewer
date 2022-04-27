#pragma once
#include "VkUtil.h"

// function which is called once at the beginning of the PCViewer which servers as indirection to speedup test compilations

void TEST(const VkUtil::Context& context);