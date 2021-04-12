#include "TransferFunctionEditor.h"

TransferFunctionEditor::TransferFunctionEditor(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool):
	device(device), physicalDevice(physicalDevice), commandPool(commandPool), queue(queue), descriptorPool(descriptorPool), editorWidth(500), editorHeight(250), previewHeight(50), transferFormat(VK_FORMAT_R8G8B8A8_UNORM), active(false), changed(false), activeChannel(3), pos({ 100, 100 }), pivot({ 0,0 })
{
    VkUtil::createImage(device, TRANSFERFUNCTIONSIZE, 1, transferFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &transferImage);
	VkMemoryRequirements memReq = {};
	vkGetImageMemoryRequirements(device, transferImage, &memReq);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
	VkResult err = vkAllocateMemory(device, &allocInfo, nullptr, &transferMemory);
	check_vk_result(err);

	vkBindImageMemory(device, transferImage, transferMemory, 0);
	VkUtil::createImageView(device, transferImage, transferFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1, &transferImageView);

	VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, 1, 1, &transferSampler);

	currentTransferFunction = getColorMap(ColorMap::standard);
	VkCommandBuffer commands;
	VkUtil::createCommandBuffer(device, commandPool, &commands);
	VkUtil::transitionImageLayout(commands, transferImage, transferFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	VkUtil::commitCommandBuffer(queue, commands);
	check_vk_result(vkQueueWaitIdle(queue));
	vkFreeCommandBuffers(device, commandPool, 1, &commands);

	updateGpuImage();

	transferDescriptorSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(transferSampler, transferImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, device, descriptorPool);
}

TransferFunctionEditor::~TransferFunctionEditor()
{
	vkFreeMemory(device, transferMemory, nullptr);
	vkDestroyImage(device, transferImage, nullptr);
	vkDestroyImageView(device, transferImageView, nullptr);
	vkDestroySampler(device, transferSampler, nullptr);
}

void TransferFunctionEditor::show()
{
	active = true;
}

void TransferFunctionEditor::draw()
{
	if (!active) return;
	ImGui::SetNextWindowPos(pos, ImGuiCond_Appearing, pivot);
	ImGui::Begin("Transfer Function Editor", nullptr, ImGuiWindowFlags_NoDocking| ImGuiWindowFlags_NoDecoration| ImGuiWindowFlags_NoResize);
	static char* channels[4]{ "Red", "Green", "Blue", "Alpha" };
	ImGui::Text("Select channel to edit:");
	for (int i = 0; i < 4; ++i) {
		if (i != 0) ImGui::SameLine();
		if (ImGui::RadioButton(channels[i], i == activeChannel)) activeChannel = i;
	}
	static int uniformValue = 0;
	ImGui::SliderInt("##uniformtransformvalue", &uniformValue, 0, 255);
	ImGui::SameLine();
	if (ImGui::Button("Set channel value")) {
		for (int i = 0; i < TRANSFERFUNCTIONSIZE; ++i) {
			currentTransferFunction[i * 4 + activeChannel] = uniformValue;
		}
		updateGpuImage();
	}
	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImVec2 canvas_size{ float(editorWidth), float(editorHeight) };
	ImDrawList* drawList = ImGui::GetWindowDrawList();
	drawList->PushClipRect(canvas_pos, canvas_pos + canvas_size);
	drawList->AddRect(canvas_pos, canvas_pos + canvas_size, ImColor(180, 180, 180, 255));
	ImGui::InvisibleButton("cnvas_invb", canvas_size);
	const ImGuiIO& io = ImGui::GetIO();
	if (ImGui::IsItemHovered() && io.MouseDown[0]) {
		ImVec2 localPos = ImGui::GetMousePos() - canvas_pos;
		uint32_t x = localPos.x / (canvas_size.x - 1) * (TRANSFERFUNCTIONSIZE - 1) + .5f;
		uint32_t y = localPos.y / canvas_size.y * (255) + .5f;
		y = 255 - y;
		currentTransferFunction[x * 4 + activeChannel] = y;
		//check for missed drags
		ImVec2 drag = ImGui::GetMouseDragDelta();
		if (std::abs(drag.x) >= .1f) {
			int startInd = (localPos.x - drag.x) / canvas_size.x * (TRANSFERFUNCTIONSIZE - 1) + .5, endInd = x;
			if (startInd > endInd) {
				std::swap(startInd, endInd);
				endInd++;
			}
			for (int i = startInd; i < endInd; ++i) {
				if (i >= currentTransferFunction.size() / 4) break;
				if (i < 0) continue;
				float alpha = (i - startInd) / (endInd - startInd - 1.0f);
				if (endInd - startInd - 1.0f <= 0)
					alpha = 0;
				float s = drag.x / std::abs(drag.x);
				currentTransferFunction[i * 4 + activeChannel] = (1 - alpha) * s * drag.y + y;
			}
		}
		ImGui::ResetMouseDragDelta();
		changed = true;
	}
	if (ImGui::IsMouseReleased(0) && changed) {
		changed = false;
		updateGpuImage();
	}

	drawList->PopClipRect();
	for (int c = 0; c < 4; ++c) {
		for (int i = 0; i < 255; ++i) {
			ImVec2 a, b;
			a = { i / 255.0f * canvas_size.x + canvas_pos.x, -currentTransferFunction[i * 4 + c] / 255.0f * canvas_size.y + canvas_pos.y + canvas_size.y};
			b = { (i + 1) / 255.0f * canvas_size.x + canvas_pos.x, -currentTransferFunction[(i + 1) * 4 + c] / 255.0f * canvas_size.y + canvas_pos.y + canvas_size.y };
			int r = (c == 0 || c == 3) ? 255 : 0;
			int g = (c == 1 || c == 3) ? 255 : 0;
			int bl = (c == 2 || c == 3) ? 255 : 0;
			
			drawList->AddLine(a, b, ImColor(r, g, bl, (c == activeChannel) ? 255 : 128));
		}
	}

	ImGui::Image((ImTextureID)transferDescriptorSet, { canvas_size.x, (float)previewHeight });

	//deactivates the window when not focues(click next to the winodw to deactivate it)
	if (!ImGui::IsWindowFocused()) active = false;
	ImGui::End();
}

std::vector<uint8_t> TransferFunctionEditor::getColorMap(ColorMap map)
{
	if (map == ColorMap::standard) {
		std::vector<uint8_t> ret(4 * TRANSFERFUNCTIONSIZE, 255);
		for (int i = 0; i < TRANSFERFUNCTIONSIZE; ++i) {
			ret[i * 4 + 3] = 128;
		}
		return ret;
	}
	return std::vector<uint8_t>();
}

VkImageView TransferFunctionEditor::getTransferImageView()
{
	return transferImageView;
}

VkDescriptorSet TransferFunctionEditor::getTransferDescriptorSet()
{
	return transferDescriptorSet;
}

void TransferFunctionEditor::setNextEditorPos(const ImVec2& pos, const ImVec2& pivot)
{
	this->pos = pos;
	this->pivot = pivot;
}

void TransferFunctionEditor::updateGpuImage()
{
	VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, transferImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, transferFormat, TRANSFERFUNCTIONSIZE, 1, 1, currentTransferFunction.data(), currentTransferFunction.size());
}
