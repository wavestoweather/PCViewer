#include "View3d.h"

View3d::View3d()
{
	descriptorSet = VK_NULL_HANDLE;
}

View3d::~View3d()
{
}

void View3d::render()
{
}

void View3d::setDescriptorSet(VkDescriptorSet descriptor)
{
	descriptorSet = descriptor;
}
