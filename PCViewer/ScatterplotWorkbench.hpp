#pragma once

#include<vulkan/vulkan.h>
#include<vector>
#include<string>
#include<map>
#include"imgui/imgui.h"
#include"imgui/imgui_impl_vulkan.h"
#include"VkUtil.h"
#include"PCUtil.h"
#include"Structures.hpp"

// workbench for scatter plots
// all scatterplots are scatterplot matrices which can be reduced to only show wanted parameter combinations
class ScatterplotWorkbench{
public:
    // the scatter plot workbench can have multiple scatter plots
    struct ScatterPlot{
        // each drawlist which is assigned to the scatterplot has stored its information in the DrawListInstance struct
        struct DrawListInstance{
            VkUtil::Context context{};
            bool active{true};
            std::string drawListName{};
            VkBuffer data{};
            VkBuffer indices{};
            VkBufferView activeData{};
            //private vulkan ressources which have to be destroyed
            struct UBO{
                float spacing{.01f};
                float radius{1};
                uint32_t showInactivePoints{1};
                uint32_t matrixSize;
                float color[4]{1,0,0,.3f};
                float inactiveColor[4]{.1f,.1f,.1f,.1f};
            }uniformBuffer;
            VkDescriptorSet descSet{};
            VkDescriptorSetLayout descSetLayout{};
            VkBuffer ubo{};
            VkDeviceMemory uboMemory{};
            uint32_t indicesSize;
            const std::vector<Attribute>& attributes;

            DrawListInstance(VkUtil::Context context, const DrawList& drawList, VkBuffer data, VkBufferView activeData, VkBuffer indices, uint32_t indicesSize, VkDescriptorSetLayout descriptorSetLayout, const std::vector<Attribute>& attributes):
            context(context),
            drawListName(drawList.name),
            data(data),
            activeData(activeData),
            indicesSize(indicesSize),
            indices(indices),
            descSetLayout(descriptorSetLayout),
            attributes(attributes)
            {
                setupUniformBuffer();
            }

            DrawListInstance(const DrawListInstance& other):
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

            DrawListInstance(DrawListInstance&& other):
            context(other.context), drawListName(other.drawListName), data(other.data), activeData(other.activeData), indicesSize(other.indicesSize), indices(other.indices),
            descSet(other.descSet), descSetLayout(other.descSetLayout), ubo(other.ubo), uboMemory(other.uboMemory),
            uniformBuffer(other.uniformBuffer), active(other.active), attributes(other.attributes)
            {
                std::cout << "move it" << std::endl;
                other.descSet = 0;
                other.ubo = 0;
                other.uboMemory = 0;
            }

            DrawListInstance operator=(const DrawListInstance& other){
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
            }

            void setupUniformBuffer(){
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

            void updateUniformBufferData(const std::vector<Attribute>& attributes){
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

            ~DrawListInstance(){
                if(descSet) vkFreeDescriptorSets(context.device, context.descriptorPool, 1, &descSet);
                if(ubo) vkDestroyBuffer(context.device, ubo, nullptr);
                if(uboMemory) vkFreeMemory(context.device, uboMemory, nullptr);
            }
        };

        int curWidth = 0, curHeight = 0;

        VkUtil::Context context;
        VkImage resultImage{};
        VkImageView resultImageView{};
        VkSampler sampler{};
        VkFramebuffer framebuffer{};
        VkDeviceMemory imageMemory{};
        VkDescriptorSet resultImageSet{};
        uint activeAttributesCount{};
        std::vector<uint8_t> activeAttributes;
        std::vector<Attribute>& attributes;
        std::vector<DrawListInstance> dls;
        float matrixSpacing{.05f};
        float matrixBorderWidth{1};
        ImVec4 matrixBorderColor{1,1,1,1};

        //external vulkan resources (dont have to be destroyed)
        VkRenderPass renderPass;
        VkDescriptorSetLayout descriptorSetLayout;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;

        ScatterPlot(VkUtil::Context context, int width, int height, VkRenderPass renderPass, VkDescriptorSetLayout descriptorSetLayout, VkPipeline pipeline, VkPipelineLayout pipelineLayout, std::vector<Attribute>& attributes): 
        context(context), 
        renderPass(renderPass),
        activeAttributesCount(attributes.size()),
        activeAttributes(attributes.size(), true), 
        attributes(attributes),
        descriptorSetLayout(descriptorSetLayout),
        pipeline(pipeline),
        pipelineLayout(pipelineLayout)
        {
            resizeImage(width, height);
        }
        ~ScatterPlot(){
            if(resultImage) vkDestroyImage(context.device, resultImage, nullptr);
            if(resultImageView) vkDestroyImageView(context.device, resultImageView, nullptr);
            if(imageMemory) vkFreeMemory(context.device, imageMemory, nullptr);
            if(framebuffer) vkDestroyFramebuffer(context.device, framebuffer, nullptr);
            if(sampler) vkDestroySampler(context.device, sampler, nullptr);
        }
        void resizeImage(int width, int height){
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

        void draw(int index){
            ImGui::BeginChild(("Scatterplot" + std::to_string(index)).c_str(),{0,0}, true);
            ImGui::Text("Attribute activations:");
            for(int i = 0; i < attributes.size(); ++i){
                if(i != 0) ImGui::SameLine();
                if(ImGui::Checkbox((attributes[i].name + "##scatter").c_str(), (bool*)&activeAttributes[i])){
                    activeAttributesCount = 0;
                    for(auto x: activeAttributes) if(x) ++activeAttributesCount;
                    updatePlot();
                }
            }
            ImGui::Text("Drawlists");
            for(DrawListInstance& dl: dls){
                if(ImGui::ArrowButton(("##ab" + dl.drawListName).c_str(), ImGuiDir_Up)){
                    int i = 0;
                    for(i = 0; i < dls.size(); ++i) if(dls[i].drawListName == dl.drawListName) break;
                    if(i != 0){
                        DrawListInstance tmp = dls[i];
                        dls[i] = dls[i - 1];
                        dls[i - 1] = tmp;
                        updatePlot();
                    }
                }
                ImGui::SameLine(25);
                if(ImGui::ArrowButton(("##abdown" + dl.drawListName).c_str(), ImGuiDir_Down)){
                    int i = 0;
                    for(i = 0; i < dls.size(); ++i) if(dls[i].drawListName == dl.drawListName) break;
                    if(i != dls.size() - 1){
                        DrawListInstance tmp = dls[i];
                        dls[i] = dls[i + 1];
                        dls[i + 1] = tmp;
                        updatePlot();
                    }
                }
                ImGui::SameLine(50);
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
            for(int i = 0; i < attributes.size() - 1; ++i){
                int curAttr = i + 1;
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
                    addDrawList(*dl);
                    updatePlot();
                }
            }
            //Drawing boxes around the matrix elements
            float curX = imagePos.x;
            curY = imagePos.y;
            for(int i = 0; i < activeAttributesCount - 1; ++i){
                for(int j = 0; j <= i; ++j){
                    ImGui::GetWindowDrawList()->AddRect({curX, curY}, {curX + xSpacing, curY + xSpacing}, ImGui::GetColorU32(matrixBorderColor), 0, ImDrawCornerFlags_All, matrixBorderWidth);
                    curX += xSpacing;
                }
                curX = imagePos.x;
                curY += xSpacing;
            }

            float curSpace = xSpacing / 2 + leftSpace;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + xSpacing / 2 + leftSpace);
            bool firstLabel = true;
            curPlace = 0;
            for(int i = 0; i < attributes.size() - 1; ++i){
                if(!activeAttributes[i] || curPlace == activeAttributesCount - 1) continue;
                if(!firstLabel) ImGui::SameLine(curSpace); 
                if(firstLabel) firstLabel = false;
                ImGui::Text(attributes[i].name.c_str());
                curSpace += xSpacing;
                ++curPlace;
            }
            
            ImGui::EndChild();
        };

        void updatePlot(){
            VkCommandBuffer commandBuffer;
            VkUtil::createCommandBuffer(context.device, context.commandPool, &commandBuffer);
            updateRender(commandBuffer);
            VkUtil::commitCommandBuffer(context.queue, commandBuffer);
            VkResult res = vkQueueWaitIdle(context.queue); check_vk_result(res);
            vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
        }

        void updateRender(VkCommandBuffer commandBuffer){
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
        };

        void addDrawList(DrawList& dl){
            dls.emplace_back(context, dl, dl.buffer, dl.activeIndicesBufferView, dl.indicesBuffer, dl.indices.size(), descriptorSetLayout, attributes);
            activeAttributes.resize(attributes.size(), 1);
            activeAttributesCount = 0;
            for(auto x: activeAttributes) if(x) ++activeAttributesCount;
        }
    };

    ScatterplotWorkbench(VkUtil::Context context, std::vector<Attribute>& attributes): context(context), attributes(attributes){
        createPipeline();
    }

    void addPlot(std::vector<Attribute>& attributes){
        scatterPlots.emplace_back(context, defaultWidth, defaultHeight, renderPass, descriptorSetLayout, pipeline, pipelineLayout, attributes);
    }

    ~ScatterplotWorkbench(){
        if(pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
        if(pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        if(descriptorSetLayout) vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);
        if(renderPass) vkDestroyRenderPass(context.device, renderPass, nullptr);
    }

    void draw(){
        if(!active) return;
        ImGui::Begin("Scatterplot Workbench", &active);
        int c = 0;
        for(ScatterPlot& s: scatterPlots){
            s.draw(c++);
        }
        if(ImGui::Button("+", ImVec2{500,0})){
            addPlot(attributes);
        }
        ImGui::End();
    }

    void updateRenders(const std::vector<int>& attrIndices){
        VkCommandBuffer commandBuffer;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commandBuffer);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        for(ScatterPlot& s: scatterPlots){
            bool change = false;
            for(int i: attrIndices) change |= s.activeAttributes[i];
            change |= attrIndices.empty();
            if(change) s.updateRender(commandBuffer);
        }
        VkUtil::commitCommandBuffer(context.queue, commandBuffer);
        VkResult res = vkQueueWaitIdle(context.queue); check_vk_result(res);
        vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
    }

    bool active = false;
    std::vector<ScatterPlot> scatterPlots;
    int defaultWidth = 800, defaultHeight = 800;
    bool showInactivePoints{true};
    struct Polygon{
        bool change{false};
        std::vector<ImVec2> borderPoints;
    };
    using Polygons = std::vector<Polygon>;
    std::map<std::string, Polygons> lassoSelections;
protected:
    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};
    VkDescriptorSetLayout descriptorSetLayout{};
    VkRenderPass renderPass{};

    VkUtil::Context context;
    std::vector<Attribute>& attributes;

    struct PushConstant{
        uint32_t posX;
        uint32_t posY;
        uint32_t xAttr;
        uint32_t yAttr;
        uint32_t discard;
    };

    void createPipeline(){
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
};