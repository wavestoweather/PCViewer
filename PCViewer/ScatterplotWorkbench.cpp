#include "ScatterplotWorkbench.hpp"
#include <cmath>

ImVec2 ScatterplotWorkbench::pixelPosToParameterPos(const ImVec2& mousePos, const ImVec2& borderMin, const ImVec2& borderMax, int attr1, int attr2, const std::vector<Attribute>& pcAttributes){
    float x = mousePos.y - borderMax.y;
    x /= borderMax.y - borderMin.y;
    x *= -1; //inverting y axis
    x *= pcAttributes[attr1].max - pcAttributes[attr1].min;
    x += pcAttributes[attr1].min;
    float y = mousePos.x - borderMin.x;
    y /= borderMax.x - borderMin.x;
    y *= pcAttributes[attr2].max - pcAttributes[attr2].min;
    y += pcAttributes[attr2].min;
    return {x, y};
}

ImVec2 ScatterplotWorkbench::parameterPosToPixelPos(const ImVec2& paramPos, const ImVec2& borderMin, const ImVec2& borderMax, int attr1, int attr2, const std::vector<Attribute>& pcAttributes){
    float x = paramPos.x - pcAttributes[attr1].min;
    x /= pcAttributes[attr1].max - pcAttributes[attr1].min;
    x *= -1;
    x *= borderMax.y - borderMin.y;
    x += borderMax.y;
    float y = paramPos.y - pcAttributes[attr2].min;
    y /= pcAttributes[attr2].max - pcAttributes[attr2].min;
    y *= borderMax.x - borderMin.x;
    y += borderMin.x;
    return {y, x};
}

float ScatterplotWorkbench::distance2(const ImVec2& a, const ImVec2& b){
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

float ScatterplotWorkbench::distance(const ImVec2& a, const ImVec2& b){
    return std::sqrt(distance2(a, b));
}

ScatterplotWorkbench::ScatterPlot::DrawListInstance::DrawListInstance(VkUtil::Context context, const DrawList& drawList, VkBuffer data, VkBufferView activeData, VkBuffer indices, uint32_t indicesSize, VkDescriptorSetLayout descriptorSetLayout, const std::vector<Attribute>& attributes):
context(context),
drawListName(drawList.name),
data(data),
activeData(activeData),
indicesSize(indicesSize),
indices(indices),
descSetLayout(descriptorSetLayout),
attributes(attributes)
{
    std::copy_n(&drawList.color.x, 4, uniformBuffer.color);
    setupUniformBuffer();
}

ScatterplotWorkbench::ScatterPlot::DrawListInstance::DrawListInstance(const DrawListInstance& other):
context(other.context),
drawListName(other.drawListName),
data(other.data),
activeData(other.activeData),
indicesSize(other.indicesSize),
indices(other.indices),
uniformBuffer(other.uniformBuffer),
active(other.active),
descSetLayout(other.descSetLayout),
attributes(other.attributes)
{
    setupUniformBuffer();
};

ScatterplotWorkbench::ScatterPlot::DrawListInstance::DrawListInstance(DrawListInstance&& other):
context(other.context), drawListName(other.drawListName), data(other.data), activeData(other.activeData), indicesSize(other.indicesSize), indices(other.indices),
descSet(other.descSet), descSetLayout(other.descSetLayout), ubo(other.ubo), uboMemory(other.uboMemory),
uniformBuffer(other.uniformBuffer), active(other.active), attributes(other.attributes)
{
    other.descSet = 0;
    other.ubo = 0;
    other.uboMemory = 0;
}

ScatterplotWorkbench::ScatterPlot::DrawListInstance& ScatterplotWorkbench::ScatterPlot::DrawListInstance::operator=(const ScatterplotWorkbench::ScatterPlot::DrawListInstance& other){
    context = other.context;
    drawListName = other.drawListName;
    data = other.data;
    activeData = other.activeData;
    indicesSize = other.indicesSize;
    indices = other.indices;
    uniformBuffer = other.uniformBuffer;
    active = other.active;
    descSetLayout = other.descSetLayout;
    //assert(attributes == other.attributes);
    
    setupUniformBuffer();
    return *this;
}

ScatterplotWorkbench::ScatterPlot::DrawListInstance& ScatterplotWorkbench::ScatterPlot::DrawListInstance::operator=(ScatterplotWorkbench::ScatterPlot::DrawListInstance&& other){
context = other.context;
    drawListName = other.drawListName;
    data = other.data;
    activeData = other.activeData;
    indicesSize = other.indicesSize;
    indices = other.indices;
    uniformBuffer = other.uniformBuffer;
    active = other.active;
    descSetLayout = other.descSetLayout;
    descSet = other.descSet;
    ubo = other.ubo;
    uboMemory = other.uboMemory;
    other.descSet = 0;
    other.ubo = 0;
    other.uboMemory = 0;
    return *this;
}

void ScatterplotWorkbench::ScatterPlot::DrawListInstance::setupUniformBuffer(){
    uint32_t uboSize = sizeof(UBO) + 2 * attributes.size() * sizeof(float);
    VkUtil::createBuffer(context.device, uboSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ubo);
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, ubo, &memReq);
    VkMemoryAllocateInfo memAlloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memAlloc.allocationSize = memReq.size;
    uint32_t memBits = memReq.memoryTypeBits;
    memAlloc.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, memBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VkResult res = vkAllocateMemory(context.device, &memAlloc, nullptr, &uboMemory); check_vk_result(res);
    vkBindBufferMemory(context.device, ubo, uboMemory, 0);

    VkUtil::createDescriptorSets(context.device, {descSetLayout}, context.descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(context.device, data, VK_WHOLE_SIZE, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateTexelBufferDescriptorSet(context.device, activeData, 1, descSet);
    VkUtil::updateDescriptorSet(context.device, indices, VK_WHOLE_SIZE, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(context.device, ubo, uboSize, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
}

void ScatterplotWorkbench::ScatterPlot::DrawListInstance::updateUniformBufferData(const std::vector<Attribute>& attributes){
    uint32_t uboSize = sizeof(UBO) + 2 * attributes.size() * sizeof(float);
    std::vector<uint8_t> uniformBytes(uboSize);
    std::copy(reinterpret_cast<uint8_t*>(&uniformBuffer), reinterpret_cast<uint8_t*>((&uniformBuffer) + 1), uniformBytes.begin());
    float* miMa = reinterpret_cast<float*>(uniformBytes.data() + sizeof(UBO));
    for(int i = 0; i < attributes.size(); ++i){
        //std::cout << reinterpret_cast<ulong>(miMa + 2 * i) << "|" << reinterpret_cast<ulong>(uniformBytes.data() + uboSize) << std::endl;
        assert(reinterpret_cast<uint8_t*>(miMa + 2 * i) < uniformBytes.data() + uboSize);
        miMa[2 * i] = attributes[i].min;
        miMa[2 * i + 1] = attributes[i].max;
    }
    VkUtil::uploadData(context.device, uboMemory, 0, uboSize, uniformBytes.data());
}

ScatterplotWorkbench::ScatterPlot::ScatterPlot(VkUtil::Context context, int width, int height, VkRenderPass renderPass, VkDescriptorSetLayout descriptorSetLayout, VkPipeline pipeline, VkPipelineLayout pipelineLayout, std::vector<Attribute>& attributes): 
context(context), 
renderPass(renderPass),
activeAttributesCount(attributes.size()),
activeAttributes( attributes.size(), true), 
attributes(attributes),
descriptorSetLayout(descriptorSetLayout),
pipeline(pipeline),
pipelineLayout(pipelineLayout),
id(scatterPlotCounter++)
{
    resizeImage(width, height);
}

ScatterplotWorkbench::ScatterPlot::~ScatterPlot(){
    if(resultImage) vkDestroyImage(context.device, resultImage, nullptr);
    if(resultImageView) vkDestroyImageView(context.device, resultImageView, nullptr);
    if(imageMemory) vkFreeMemory(context.device, imageMemory, nullptr);
    if(framebuffer) vkDestroyFramebuffer(context.device, framebuffer, nullptr);
    if(sampler) vkDestroySampler(context.device, sampler, nullptr);
}

void ScatterplotWorkbench::ScatterPlot::resizeImage(int width, int height){
    VkResult res;
    if(!sampler){
        VkUtil::createImageSampler(context.device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 1, 1, &sampler);
    }
    if(width != curWidth || height != curHeight){
        curWidth = width;
        curHeight = height;
        if(resultImage) vkDestroyImage(context.device, resultImage, nullptr);
        if(resultImageView) vkDestroyImageView(context.device, resultImageView, nullptr);
        if(imageMemory) vkFreeMemory(context.device, imageMemory, nullptr);
        if(framebuffer) vkDestroyFramebuffer(context.device, framebuffer, nullptr);
        VkUtil::createImage(context.device, width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &resultImage);
        
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(context.device, resultImage, &memReq);
        VkMemoryAllocateInfo memAlloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        memAlloc.allocationSize = memReq.size;
        uint32_t memBits = memReq.memoryTypeBits;
        memAlloc.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, memBits, 0);
        res = vkAllocateMemory(context.device, &memAlloc, nullptr, &imageMemory); check_vk_result(res);
        vkBindImageMemory(context.device, resultImage, imageMemory, 0);
        VkUtil::createImageView(context.device, resultImage, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &resultImageView);
        
        VkUtil::createFrameBuffer(context.device, renderPass, {resultImageView}, width, height, &framebuffer);
        //transform image layout form undefined to shader read only optimal
        VkCommandBuffer command;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &command);
        VkUtil::transitionImageLayout(command, resultImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkUtil::commitCommandBuffer(context.queue, command);
        res = vkQueueWaitIdle(context.queue);
        vkFreeCommandBuffers(context.device, context.commandPool, 1, &command);
        if(resultImageSet) vkFreeDescriptorSets(context.device, context.descriptorPool, 1, &resultImageSet);
        resultImageSet = static_cast<VkDescriptorSet>(ImGui_ImplVulkan_AddTexture(sampler, resultImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, context.device, context.descriptorPool));
    }
}

void ScatterplotWorkbench::ScatterPlot::draw(int index){
    ImGui::BeginChild(("Scatterplot" + std::to_string(index)).c_str(),{0,0}, true);
    ImGui::Text("Attribute activations:");
    for(int i = 0; i < activeAttributes.size(); ++i){
        if(i != 0) ImGui::SameLine();
        if(ImGui::Checkbox((attributes[i].name + "##scatter").c_str(), (bool*)&activeAttributes[i])){
            activeAttributesCount = 0;
            for(auto x: activeAttributes) if(x) ++activeAttributesCount;
            updatePlot();
        }
    }
    ImGui::Text("Drawlists");
    for(auto dlI = dls.rbegin(); dlI != dls.rend(); ++dlI){
        DrawListInstance& dl = *dlI;
        if(ImGui::ArrowButton(("##ab" + dl.drawListName).c_str(), ImGuiDir_Up)){
            int i = 0;
            for(i = 0; i < dls.size(); ++i) if(dls[i].drawListName == dl.drawListName) break;
            if(i != dls.size() - 1){
                DrawListInstance tmp = std::move(dls[i]);
                dls[i] = std::move(dls[i + 1]);
                dls[i + 1] = std::move(tmp);
                updatePlot();
            }
        }
        ImGui::SameLine(30);
        if(ImGui::ArrowButton(("##abdown" + dl.drawListName).c_str(), ImGuiDir_Down)){
            int i = 0;
            for(i = 0; i < dls.size(); ++i) if(dls[i].drawListName == dl.drawListName) break;
            if(i != 0){
                DrawListInstance tmp = std::move(dls[i]);
                dls[i] = std::move(dls[i - 1]);
                dls[i - 1] = std::move(tmp);
                updatePlot();
            }
        }
        ImGui::SameLine(55);
        if(ImGui::Checkbox((dl.drawListName + "##scatter").c_str(), &dl.active)){
            updatePlot();
        }
        ImGui::SameLine(200);
        if(ImGui::ColorEdit4(("Color##scatter"+ dl.drawListName).c_str(), dl.uniformBuffer.color, ImGuiColorEditFlags_NoInputs)){
            dl.updateUniformBufferData(attributes);
            updatePlot();
        }
        ImGui::SameLine(300);
        if(ImGui::Checkbox(("Deactivated Lines##" + dl.drawListName).c_str(), (bool*)&dl.uniformBuffer.showInactivePoints)){
            dl.updateUniformBufferData(attributes);
            updatePlot();
        }
        ImGui::SameLine(500);
        if(ImGui::ColorEdit4(("ColorInactive##scatter"+ dl.drawListName).c_str(), dl.uniformBuffer.inactiveColor, ImGuiColorEditFlags_NoInputs)){
            dl.updateUniformBufferData(attributes);
            updatePlot();
        }
        ImGui::SameLine(600);
        ImGui::PushItemWidth(100);
        const static char* pointTypes[]{"Circle", "Square"};
        if(ImGui::BeginCombo(("##PoFo" + dl.drawListName).c_str(), pointTypes[dl.uniformBuffer.showInactivePoints >> 1])){
            for(int i = 0; i < 2; ++i){
                if(ImGui::MenuItem(pointTypes[i])) {i ? dl.uniformBuffer.showInactivePoints |= i << 1 : dl.uniformBuffer.showInactivePoints ^= dl.uniformBuffer.showInactivePoints & 2;}
                updatePlot();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine(750);
        if(ImGui::SliderFloat(("Radius##scatter"+dl.drawListName).c_str(), &dl.uniformBuffer.radius, 1, 20)){
            dl.updateUniformBufferData(attributes);
            updatePlot();
        }
        ImGui::PopItemWidth();
    }
    //Plot section
    ImGui::Separator();
    //drawing the labels on the left
    float curY = ImGui::GetCursorPosY();
    float xSpacing = curWidth / (activeAttributesCount - 1);
    const int leftSpace = 150;
    int curPlace = 0;
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + xSpacing / 2);
    int startInd = 1; while(activeAttributes.size() && !activeAttributes[startInd - 1]) ++startInd;
    for(int i = startInd; activeAttributes.size() && i < attributes.size(); ++i){
        int curAttr = i;
        if(!activeAttributes[curAttr] || curPlace == activeAttributesCount - 1) continue;
        ImGui::Text(attributes[curAttr].name.c_str());
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + xSpacing - ImGui::GetTextLineHeightWithSpacing());
        ++curPlace;
    }
    ImGui::SetCursorPosY(curY);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + leftSpace);
    ImVec2 imagePos = ImGui::GetCursorScreenPos();
    ImVec2 imageSize{curWidth, curHeight};
    ImGui::Image(static_cast<ImTextureID>(resultImageSet), ImVec2{curWidth, curHeight});
    if(ImGui::BeginDragDropTarget()){
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")){
            DrawList* dl = *((DrawList**)payload->Data);
            addDrawList(*dl, attributes);
            updatePlot();
        }
    }
    //Drawing boxes around the matrix elements
    //Drawing the lasso selections
    float curX = imagePos.x;
    int curAttr = 1; while(activeAttributes.size() && !activeAttributes[curAttr - 1]) ++curAttr;
    curY = imagePos.y;
    for(int i = 0; activeAttributesCount && i < activeAttributesCount - 1; ++i){
        while(!activeAttributes[curAttr] && curAttr < activeAttributes.size()) ++curAttr;
        int curAttr2 = 0;
        for(int j = 0; j <= i; ++j){
            //boxes
            ImGui::GetWindowDrawList()->AddRect({curX, curY}, {curX + xSpacing, curY + xSpacing}, ImGui::GetColorU32(matrixBorderColor), 0, ImDrawCornerFlags_All, matrixBorderWidth);
            //Lassos
            while(!activeAttributes[curAttr2] && curAttr2 < activeAttributes.size()) ++curAttr2;
            if(dls.size() && lassoSelections.find(dls.front().drawListName) != lassoSelections.end()){
                auto lasso = std::find_if(lassoSelections[dls.front().drawListName].begin(), lassoSelections[dls.front().drawListName].end(),
                                            [&](const Polygon& polygon){return polygon.attr1 == curAttr && polygon.attr2 == curAttr2;});
                if(lasso != lassoSelections[dls.front().drawListName].end()){
                    for(int p = 1; p < lasso->borderPoints.size(); ++p){
                        ImVec2 a = parameterPosToPixelPos(lasso->borderPoints[p - 1], {curX, curY}, {curX + xSpacing, curY + xSpacing}, curAttr, curAttr2, attributes);
                        ImVec2 b = parameterPosToPixelPos(lasso->borderPoints[p], {curX, curY}, {curX + xSpacing, curY + xSpacing}, curAttr, curAttr2, attributes);
                        ImGui::GetWindowDrawList()->AddLine(a, b, ImGui::GetColorU32({0,0,1,1}), 2);
                    }
                    ImVec2 a = parameterPosToPixelPos(lasso->borderPoints[0], {curX, curY}, {curX + xSpacing, curY + xSpacing}, curAttr, curAttr2, attributes);
                    ImVec2 b = parameterPosToPixelPos(lasso->borderPoints.back(), {curX, curY}, {curX + xSpacing, curY + xSpacing}, curAttr, curAttr2, attributes);
                    ImGui::GetWindowDrawList()->AddLine(a, b, ImGui::GetColorU32({0,0,1,1}), 2);
                }
            }
            curX += xSpacing;
            ++curAttr2;
        }
        curX = imagePos.x;
        curY += xSpacing;
        ++curAttr;
    }
    //lasso selection
    ImVec2 mousePos = ImGui::GetMousePos();
    bool inside = mousePos.x > imagePos.x && mousePos.x < imagePos.x + imageSize.x
                    && mousePos.y > imagePos.y && mousePos.y < imagePos.y + imageSize.y;
    static int attr1 = -1, attr2 = -1, plotId = -1;
    static ImVec2 borderMin, borderMax;
    static ScatterPlot* curPlot{};
    static ImVec2 prevPointPos;
    if(ImGui::IsMouseClicked(0) && inside && attr1 < 0){   //begin lasso selection. Get attributes
        plotId = id;
        curPlot = this;
        curX = imagePos.x;
        curY = imagePos.y;
        bool done = false;
        attr1 = 1; attr2 = 0; while(activeAttributes.size() && !activeAttributes[attr1 - 1]) ++attr1;
        for(int i = 0; activeAttributesCount && i < activeAttributesCount - 1; ++i){
            while(!activeAttributes[attr1] && attr1 < activeAttributes.size()) ++attr1;
            for(int j = 0; j <= i && !done; ++j){
                while(!activeAttributes[attr2] && attr2 < activeAttributes.size()) ++attr2;
                inside = mousePos.x > curX && mousePos.x < curX + xSpacing
                    && mousePos.y > curY && mousePos.y < curY + xSpacing;
                if(inside) {
                    done = true;
                    borderMin = {curX, curY};    // y is already inverted here for easier later calculations
                    borderMax = {curX + xSpacing, curY + xSpacing};
                    break;
                }
                curX += xSpacing;
                ++attr2;
            }
            if(done) break;
            curX = imagePos.x;
            curY += xSpacing;
            attr2 = 0;
            ++attr1;
        }
        //std::cout << "Attr1: " << attributes[attr1].name << " | Attr2:" << attributes[attr2].name << std::endl; 
        bool lassoCreated = false;
        if(dls.size() && lassoSelections.find(dls.front().drawListName) != lassoSelections.end()){
            auto p = std::find_if(lassoSelections.find(dls.front().drawListName)->second.begin(), lassoSelections.find(dls.front().drawListName)->second.end(), [&](Polygon& p){return p.attr1 == attr1 && p.attr2 == attr2;});
            if(p != lassoSelections.find(dls.front().drawListName)->second.end()){
                *p = {attr1, attr2, {}};
                std::swap(lassoSelections.find(dls.front().drawListName)->second.back(), *p);
                lassoCreated = true;
            }
        }
        if(dls.size() && lassoSelections.find(dls.front().drawListName) == lassoSelections.end()){
            lassoSelections[dls.front().drawListName] = {{attr1, attr2, {}}};
            lassoCreated = true;
        }
        if(dls.size() && !lassoCreated)
            lassoSelections[dls.front().drawListName].push_back({attr1, attr2, {}});
    }
    //section to continously handle lasso creation
    if(plotId == id){
        if(!ImGui::IsMouseDown(0)){  //stop lasso
            attr1 = -1; attr2 = -1; plotId = -1;
            updatedDrawlists.push_back(dls.front().drawListName);
            auto& points = lassoSelections[dls.front().drawListName].back().borderPoints;
            if(points.size() < 3){  //deleting the brush
                lassoSelections[dls.front().drawListName].pop_back();
            }
        }
        else{
            assert(lassoSelections[dls.front().drawListName].back().attr1 == attr1 && lassoSelections[dls.front().drawListName].back().attr2 == attr2);
            auto& points = lassoSelections[dls.front().drawListName].back().borderPoints;
            if(points.empty()){ //first point
                points.push_back(pixelPosToParameterPos(mousePos, borderMin, borderMax, attr1, attr2, attributes));
                prevPointPos = mousePos;
            }
            else if(distance2(mousePos, prevPointPos) > 25){    //on high enough distnace set next lasso point
                points.push_back(pixelPosToParameterPos(mousePos, borderMin, borderMax, attr1, attr2, attributes));
                //std::cout << attributes[attr1].name << ": " << points.back().x << std::endl;
                prevPointPos = mousePos;
            }
        }
    }
    float curSpace = xSpacing / 2 + leftSpace;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + xSpacing / 2 + leftSpace);
    bool firstLabel = true;
    curPlace = 0;
    for(int i = 0; activeAttributes.size() && i < attributes.size() - 1; ++i){
        if(!activeAttributes[i] || curPlace == activeAttributesCount - 1) continue;
        if(!firstLabel) ImGui::SameLine(curSpace); 
        if(firstLabel) firstLabel = false;
        ImGui::Text(attributes[i].name.c_str());
        curSpace += xSpacing;
        ++curPlace;
    }
    
    ImGui::EndChild();
};

void ScatterplotWorkbench::ScatterPlot::updatePlot(){
    VkCommandBuffer commandBuffer;
    VkUtil::createCommandBuffer(context.device, context.commandPool, &commandBuffer);
    updateRender(commandBuffer);
    VkUtil::commitCommandBuffer(context.queue, commandBuffer);
    VkResult res = vkQueueWaitIdle(context.queue); check_vk_result(res);
    vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
}

void ScatterplotWorkbench::ScatterPlot::updateRender(VkCommandBuffer commandBuffer){
    uint32_t matrixSize = 0;
    for(uint8_t b: activeAttributes) if(b) ++matrixSize;
    VkUtil::transitionImageLayout(commandBuffer, resultImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkUtil::beginRenderPass(commandBuffer, {{0,0,0,1}}, renderPass, framebuffer, {uint32_t(curWidth), uint32_t(curHeight)});
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkViewport viewport{0, 0, curWidth, curHeight, 0,1};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor{{0,0}, {curWidth, curHeight}};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    for(DrawListInstance& dl: dls){
        if(!dl.active) continue;
        //update descriptor set values
        dl.uniformBuffer.spacing = matrixSpacing;
        dl.uniformBuffer.matrixSize = matrixSize;
        dl.updateUniformBufferData(attributes);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &dl.descSet, 0, nullptr);
        //draw inactive points
        int posX = 0, posY = matrixSize - 2;
        if(dl.uniformBuffer.showInactivePoints){
            for(int i = 0; i < attributes.size(); ++i){
                if(!activeAttributes[i]) continue;
                for(int j = attributes.size() - 1; j > i; --j){
                    if(!activeAttributes[j]) continue;
                    PushConstant pc{posX, posY, i, j, 1};
                    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &pc);
                    vkCmdDraw(commandBuffer, dl.indicesSize, 1, 0, 0);
                    --posY;
                }
                posY = matrixSize - 2;
                ++posX;
            }
        }
        //draw active points
        posX = 0;
        posY = matrixSize - 2;
        for(int i = 0; i < attributes.size(); ++i){
            if(!activeAttributes[i]) continue;
            for(int j = attributes.size() - 1; j > i; --j){
                if(!activeAttributes[j]) continue;
                PushConstant pc{posX, posY, i, j, 2};
                vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &pc);
                vkCmdDraw(commandBuffer, dl.indicesSize, 1, 0, 0);
                --posY;
            }
            posY = matrixSize - 2;
            ++posX;
        }
    }
    vkCmdEndRenderPass(commandBuffer);  //finish render pass
    VkUtil::transitionImageLayout(commandBuffer, resultImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void ScatterplotWorkbench::ScatterPlot::addDrawList(DrawList& dl, std::vector<Attribute>& attr){
    dls.emplace_back(context, dl, dl.buffer, dl.activeIndicesBufferView, dl.indicesBuffer, dl.indices.size(), descriptorSetLayout, attributes);
    activeAttributes.resize(attributes.size(), 1);
    activeAttributesCount = 0;
    for(auto x: activeAttributes) if(x) ++activeAttributesCount;
}

void ScatterplotWorkbench::createPipeline(){
    VkShaderModule shaderModules[5] = {};
	//the vertex shader for the pipeline
	std::vector<char> vertexBytes = PCUtil::readByteFile("shader/scatter.vert.spv");
	shaderModules[0] = VkUtil::createShaderModule(context.device, vertexBytes);
	//the fragment shader for the pipeline
	std::vector<char> fragmentBytes = PCUtil::readByteFile("shader/scatter.frag.spv");
	shaderModules[4] = VkUtil::createShaderModule(context.device, fragmentBytes);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.pVertexBindingDescriptions = nullptr;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	vertexInputInfo.pVertexAttributeDescriptions = nullptr;

	//vector with the dynamic states
	std::vector<VkDynamicState> dynamicStates;
	dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);

	//Rasterizer Info
	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;

	//multisampling info
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

	//blendInfo
	VkUtil::BlendInfo blendInfo;

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.minDepthBounds = 0;
	depthStencil.maxDepthBounds = 1.0f;

	//creating the descriptor set layout
	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 2;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 3;
    bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(context.device, bindings, &descriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(descriptorSetLayout);

	VkUtil::createRenderPass(context.device, VkUtil::PASS_TYPE_COLOR_OFFLINE, &renderPass);
    std::vector<VkPushConstantRange> pushConstantRanges{{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant)}};
    VkUtil::createPipeline(context.device, &vertexInputInfo, 100, 100, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, &depthStencil, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline, pushConstantRanges);
}

std::map<std::string, Polygons> ScatterplotWorkbench::lassoSelections = {};
std::vector<std::string> ScatterplotWorkbench::updatedDrawlists{};
int ScatterplotWorkbench::scatterPlotCounter = 0;