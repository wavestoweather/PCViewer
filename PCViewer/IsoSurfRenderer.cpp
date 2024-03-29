#include "IsoSurfRenderer.h"

char IsoSurfRenderer::vertPath[]= "shader/isoSurfVert.spv";
char IsoSurfRenderer::fragPath[]= "shader/isoSurfFrag.spv";
char IsoSurfRenderer::computePath[] = "shader/isoSurfComp.spv";
char IsoSurfRenderer::activeIndComputePath[] = "shader/isoSurfActiveIndComp.spv";
char IsoSurfRenderer::binaryComputePath[] = "shader/isoSurfBinComp.spv";
char IsoSurfRenderer::binarySmoothPath[] = "shader/isoSurfSmooth.spv";
char IsoSurfRenderer::binaryCopyOnesPath[] = "shader/isoSurfCopyOnes.spv";

IsoSurfRenderer::IsoSurfRenderer(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool)
{
    imageHeight = 0;
    imageWidth = 0;
    this->device = device;
    this->physicalDevice = physicalDevice;
    this->commandPool = commandPool;
    this->queue = queue;
    this->descriptorPool = descriptorPool;
    imageMemory = VK_NULL_HANDLE;
    image =    VK_NULL_HANDLE;
    imageView =    VK_NULL_HANDLE;
    sampler = VK_NULL_HANDLE;
    image3dMemory = VK_NULL_HANDLE;
    descriptorSetLayout = VK_NULL_HANDLE,
    descriptorSet = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE; 
    pipelineLayout = VK_NULL_HANDLE;
    constantMemory = VK_NULL_HANDLE;
    vertexBuffer = VK_NULL_HANDLE;
    indexBuffer = VK_NULL_HANDLE;
    uniformBuffer = VK_NULL_HANDLE;
    commandBuffer = VK_NULL_HANDLE;
    prepareImageCommand = VK_NULL_HANDLE;
    frameBuffer = VK_NULL_HANDLE;
    imageDescriptorSet = VK_NULL_HANDLE;
    computePipeline = VK_NULL_HANDLE;
    computePipelineLayout = VK_NULL_HANDLE;
    computeDescriptorSetLayout = VK_NULL_HANDLE;
    binaryComputePipeline = VK_NULL_HANDLE;
    binaryComputePipelineLayout = VK_NULL_HANDLE;
    binaryComputeDescriptorSetLayout = VK_NULL_HANDLE;
    binaryImageSampler = VK_NULL_HANDLE;
    brushBuffer = VK_NULL_HANDLE;
    brushMemory = VK_NULL_HANDLE;
    binarySmoothPipeline = VK_NULL_HANDLE;
    binarySmoothPipelineLayout = VK_NULL_HANDLE;
    binarySmoothDescriptorSetLayout = VK_NULL_HANDLE;
    activeIndComputePipeline = VK_NULL_HANDLE;
    activeIndComputePipelineLayout = VK_NULL_HANDLE;
    activeIndComputeDescriptorSetLayout = VK_NULL_HANDLE;
    binaryCopyOnesDescriptorSetLayout = VK_NULL_HANDLE;
    binaryCopyOnesPipeline = VK_NULL_HANDLE;
    dimensionCorrectionMemory = VK_NULL_HANDLE;
    binaryCopyOnesPipelineLayout = VK_NULL_HANDLE;
    dimensionCorrectionImages[0] = VK_NULL_HANDLE;
    dimensionCorrectionImages[1] = VK_NULL_HANDLE;
    dimensionCorrectionImages[2] = VK_NULL_HANDLE;
    dimensionCorrectionViews = std::vector<VkImageView>(3, VK_NULL_HANDLE);
    brushByteSize = 0;
    shade = true;
    stepSize = .006f;

    cameraPos = glm::vec3(1, 0, 1);
    cameraRot = glm::vec2(0, .78f);
    flySpeed = .5f;
    fastFlyMultiplier = 2.5f;
    rotationSpeed = .15f;
    lightDir = glm::vec3(-1, -1, -1);
    imageBackground = { .0f,.0f,.0f,1 };

    VkPhysicalDeviceProperties devProp;
    vkGetPhysicalDeviceProperties(physicalDevice, &devProp);
    uboAlignment = devProp.limits.minUniformBufferOffsetAlignment;

    //setting up graphic resources
    
    createBuffer();
    createPipeline();
    createDescriptorSets();
    resize(width, height);

    
    const int w = 100, h = 5, de = 1;
    glm::vec4 d[w * h * de] = {};
    d[0] = glm::vec4(1, 0, 0, 1);
    d[1] = glm::vec4(1, 0, 0, 1);
    d[2] = glm::vec4(1, 0, 0, 1);
    d[3] = glm::vec4(1, 0, 0, 1);
    d[8] = glm::vec4(0, 1, 0, .5f);
    d[26] = glm::vec4(0, 0, 1, .1f);
    /*for (int i = 1; i < 27; i+=3) {
        d[4 * i] = i / 27.0f;
        d[4 * i + 1] = 1 - (i / 27.0f);
        d[4 * i + 2] = 0;
        d[4 * i + 3] = .1f;
    }*/
    //update3dImage(w, h, de, (float*)d);
    resizeBox(1.5f, 1, 1.5f);
}

IsoSurfRenderer::~IsoSurfRenderer()
{
    if (imageMemory) {
        vkFreeMemory(device, imageMemory, nullptr);
    }
    if (image) {
        vkDestroyImage(device, image, nullptr);
    }
    if (imageView) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    if (sampler) {
        vkDestroySampler(device, sampler, nullptr);
    }
    if (image3dMemory) {
        vkFreeMemory(device, image3dMemory, nullptr);
    }
    for (int i = 0; i < image3d.size(); ++i) {
        if (image3d[i]) {
            vkDestroyImage(device, image3d[i], nullptr);
        }
        if (image3dView[i]) {
            vkDestroyImageView(device, image3dView[i], nullptr);
        }
        if (image3dSampler[i]) {
            vkDestroySampler(device, image3dSampler[i], nullptr);
        }
    }
    if (frameBuffer) {
        vkDestroyFramebuffer(device, frameBuffer, nullptr);
    }
    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }
    if (pipeline) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    if (pipelineLayout) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }
    if (renderPass) {
        vkDestroyRenderPass(device, renderPass, nullptr);
    }
    if (constantMemory) {
        vkFreeMemory(device, constantMemory, nullptr);
    }
    if (vertexBuffer) {
        vkDestroyBuffer(device, vertexBuffer, nullptr);
    }
    if (indexBuffer) {
        vkDestroyBuffer(device, indexBuffer, nullptr);
    }
    if (uniformBuffer) {
        vkDestroyBuffer(device, uniformBuffer, nullptr);
    }
    if (computePipeline) {
        vkDestroyPipeline(device, computePipeline, nullptr);
    }
    if (computePipelineLayout) {
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
    }
    if (computeDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
    }
    if (binaryComputePipeline) {
        vkDestroyPipeline(device, binaryComputePipeline, nullptr);
    }
    if (binaryComputePipelineLayout) {
        vkDestroyPipelineLayout(device, binaryComputePipelineLayout, nullptr);
    }
    if (binaryComputeDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, binaryComputeDescriptorSetLayout, nullptr);
    }
    if (brushBuffer) {
        vkDestroyBuffer(device, brushBuffer, nullptr);
    }
    if (brushMemory) {
        vkFreeMemory(device, brushMemory, nullptr);
    }
    if (binaryImage.size()) {
        for (auto i : binaryImage) 
            vkDestroyImage(device, i, nullptr);
    }
    if (binaryImageMemory.size()) {
        for (auto i : binaryImageMemory)
            vkFreeMemory(device, i, nullptr);
    }
    if (binaryImageView.size()) {
        for(auto i : binaryImageView)
            vkDestroyImageView(device, i, nullptr);
    }
    if (binarySmooth.size()) {
        for (auto i : binarySmooth) {
            vkDestroyImage(device, i, nullptr);
        }
    }
    if (binarySmoothView.size()) {
        for (auto i : binarySmoothView) {
            vkDestroyImageView(device, i, nullptr);
        }
    }
    if (binaryImageSampler) {
        vkDestroySampler(device, binaryImageSampler, nullptr);
    }
    if (binarySmoothPipeline) {
        vkDestroyPipeline(device, binarySmoothPipeline, nullptr);
    }
    if (binarySmoothPipelineLayout) {
        vkDestroyPipelineLayout(device, binarySmoothPipelineLayout, nullptr);
    }
    if (binarySmoothDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, binarySmoothDescriptorSetLayout, nullptr);
    }
    if (activeIndComputePipeline) {
        vkDestroyPipeline(device, activeIndComputePipeline, nullptr);
    }
    if (activeIndComputePipelineLayout) {
        vkDestroyPipelineLayout(device, activeIndComputePipelineLayout, nullptr);
    }
    if (activeIndComputeDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, activeIndComputeDescriptorSetLayout, nullptr);
    }
    if (binaryCopyOnesDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, binaryCopyOnesDescriptorSetLayout, nullptr);
    }
    if (binaryCopyOnesPipelineLayout) {
        vkDestroyPipelineLayout(device, binaryCopyOnesPipelineLayout, nullptr);
    }
    if (binaryCopyOnesPipeline) {
        vkDestroyPipeline(device, binaryCopyOnesPipeline, nullptr);
    }
    for (auto& col : brushColors) {
        delete[] col.second;
    }
    if (dimensionCorrectionMemory) {
        vkFreeMemory(device, dimensionCorrectionMemory, nullptr);
    }
    if (dimensionCorrectionImages[0]) {
        vkDestroyImage(device, dimensionCorrectionImages[0], nullptr);
        vkDestroyImage(device, dimensionCorrectionImages[1], nullptr);
        vkDestroyImage(device, dimensionCorrectionImages[2], nullptr);
        vkDestroyImageView(device, dimensionCorrectionViews[0], nullptr);
        vkDestroyImageView(device, dimensionCorrectionViews[1], nullptr);
        vkDestroyImageView(device, dimensionCorrectionViews[2], nullptr);
    }
}

void IsoSurfRenderer::resize(uint32_t width, uint32_t height)
{
    if (imageWidth == width && imageHeight == height) {
        return;
    }
    imageWidth = width;
    imageHeight = height;
    
    check_vk_result(vkDeviceWaitIdle(device));

    if (image) {
        vkDestroyImage(device, image, nullptr);
    }
    if (imageView) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    if (frameBuffer) {
        vkDestroyFramebuffer(device, frameBuffer, nullptr);
    }
    if (sampler) {
        vkDestroySampler(device, sampler, nullptr);
    }

    createImageResources();

    //transforming the image to the right format
    createPrepareImageCommandBuffer();
    VkResult err;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &prepareImageCommand;

    err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    check_vk_result(err);

    if (image3dSampler.size()) {
        updateCommandBuffer();
    }
}

void IsoSurfRenderer::resizeBox(float width, float height, float depth)
{
    boxWidth = width;
    boxHeight = height;
    boxDepth = depth;
    render();
}

bool IsoSurfRenderer::update3dBinaryVolume(uint32_t width, uint32_t height, uint32_t depth, uint32_t amtOfAttributes, const std::vector<uint32_t>& densityAttributes, std::vector<std::pair<float, float>>& densityAttributesMinMax, glm::uvec3& positionIndices, std::vector<float*>& data, std::vector<uint32_t>& indices, std::vector<std::vector<std::pair<float,float>>>& brush, int index)
{
    int w = SpacialData::rlatSize;
    int d = SpacialData::rlonSize;
    int h = SpacialData::altitudeSize + 22;    //the top 22 layer of the dataset are twice the size of the rest
    width = w;
    height = h;
    depth = d;

    if (binaryImage.size() && index != -1 && (drawlistBrushes[index].gridDimensions[0] != width || drawlistBrushes[index].gridDimensions[1] != height || drawlistBrushes[index].gridDimensions[2] != depth)) return false;

    VkResult err;

    if (!binaryImageSampler) {
        VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 1, 1, &binaryImageSampler);
    }

    uint32_t required3dImages = densityAttributes.size();

    if (index == -1)
        posIndices.push_back(positionIndices);
    else {
        positionIndices = posIndices[index];
    }
    densityAttributesMinMax[positionIndices.x] = { SpacialData::rlat[0],SpacialData::altitude[SpacialData::rlatSize - 1] };
    densityAttributesMinMax[positionIndices.y] = { SpacialData::rlon[0],SpacialData::altitude[SpacialData::rlonSize - 1] };
    densityAttributesMinMax[positionIndices.z] = { SpacialData::altitude[0],SpacialData::altitude[SpacialData::altitudeSize - 1] };

    //destroying old resources
    if (image3dMemory) {
        vkFreeMemory(device, image3dMemory, nullptr);
    }
    for (int i = 0; i < image3d.size(); ++i) {
        if (image3d[i]) {
            vkDestroyImage(device, image3d[i], nullptr);
            image3d[i] = VK_NULL_HANDLE;
        }
        if (image3dView[i]) {
            vkDestroyImageView(device, image3dView[i], nullptr);
            image3dView[i] = VK_NULL_HANDLE;
        }
    }
    image3d.clear();
    image3dView.clear();
    image3dOffsets.clear();
    for (int i = image3dSampler.size(); i < required3dImages; ++i) {
        image3dSampler.push_back({});
        VkUtil::createImageSampler(device,VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,VK_FILTER_LINEAR,1,1,&image3dSampler.back());
    }

    //creating new resources
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    uint32_t memoryTypeBits = 0;
    VkMemoryRequirements memRequirements;
    for (int i = 0; i < required3dImages; ++i) {
        image3d.push_back({});
        image3dView.push_back({});
        image3dOffsets.push_back(0);

        image3dOffsets[i] = allocInfo.allocationSize;
        VkUtil::create3dImage(device, w, h, d, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &image3d[i]);

        vkGetImageMemoryRequirements(device, image3d[i], &memRequirements);

        allocInfo.allocationSize += memRequirements.size;
        memoryTypeBits |= memRequirements.memoryTypeBits;
    }

    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, 0);
    err = vkAllocateMemory(device, &allocInfo, nullptr, &image3dMemory);
    check_vk_result(err);
    VkCommandBuffer imageCommands;
    VkUtil::createCommandBuffer(device, commandPool, &imageCommands);
    VkClearColorValue clear = { 0,0,0,0 };
    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1 };
    for (int i = 0; i < required3dImages; ++i) {
        vkBindImageMemory(device, image3d[i], image3dMemory, image3dOffsets[i]);

        VkUtil::create3dImageView(device, image3d[i], VK_FORMAT_R32_SFLOAT, 1, &image3dView[i]);

        VkUtil::transitionImageLayout(imageCommands, image3d[i], VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdClearColorImage(imageCommands, image3d[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
        //VkUtil::transitionImageLayout(imageCommands, image3d[i], VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }
    VkUtil::commitCommandBuffer(queue, imageCommands);
    err = vkQueueWaitIdle(queue);
    check_vk_result(err);
    vkFreeCommandBuffers(device, commandPool, 1, &imageCommands);

    //checking values in the first 3d image
    //float* im = new float[width * depth * height];
    //This was checked and is working!!!!!!!
    //VkUtil::downloadImageData(device, physicalDevice, commandPool, queue, image3d[4], width, height, depth, im, width * depth * height * sizeof(float));
    //uint32_t zeroCount = 0;
    //for (int i = 0; i < width * depth * height; ++i) {
    //    if (im[i] != 0) zeroCount++;
    //}

    //std::vector<VkDescriptorSetLayout> layouts;
    //layouts.push_back(descriptorSetLayout);
    //if (!descriptorSet) {
    //    VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);
    //}
    //
    ////creating the density images via the compute pipeline ----------------------------------------
    //VkBuffer infos;
    //VkDeviceMemory infosMem;
    //uint32_t infosByteSize = sizeof(ComputeInfos) + densityAttributes.size() * sizeof(float);
    //ComputeInfos* infoBytes = (ComputeInfos*)new char[infosByteSize];
    //VkUtil::createBuffer(device, infosByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &infos);
    //allocInfo = {};
    //allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    //VkMemoryRequirements memReq = {};
    //vkGetBufferMemoryRequirements(device, infos, &memReq);
    //allocInfo.allocationSize = memReq.size;
    //allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    //vkAllocateMemory(device, &allocInfo, nullptr, &infosMem);
    //vkBindBufferMemory(device, infos, infosMem, 0);
    //
    ////fill infoBytes and upload it
    //infoBytes->amtOfAttributes = amtOfAttributes;
    //infoBytes->amtOfBrushAttributes = densityAttributes.size();
    //infoBytes->amtOfIndices = amtOfIndices;
    //infoBytes->dimX = width;
    //infoBytes->dimY = height;
    //infoBytes->dimZ = depth;
    //infoBytes->xInd = positionIndices.x;
    //infoBytes->yInd = positionIndices.y;
    //infoBytes->zInd = positionIndices.z;
    //infoBytes->xMin = densityAttributesMinMax[positionIndices.x].first;
    //infoBytes->xMax = densityAttributesMinMax[positionIndices.x].second;
    //infoBytes->yMin = densityAttributesMinMax[positionIndices.y].first;
    //infoBytes->yMax = densityAttributesMinMax[positionIndices.y].second;
    //infoBytes->zMin = densityAttributesMinMax[positionIndices.z].first;
    //infoBytes->zMax = densityAttributesMinMax[positionIndices.z].second;
    //int* inf = (int*)(infoBytes + 1);
    //for (int i = 0; i < densityAttributes.size(); ++i) {
    //    inf[i] = densityAttributes[i];
    //    //inf[3 * i + 1] = densityAttributesMinMax[i].first;
    //    //inf[3 * i + 2] = densityAttributesMinMax[i].second;
    //}
    //PCUtil::numdump((int*)(infoBytes), densityAttributes.size() + 16);
    //VkUtil::uploadData(device, infosMem, 0, infosByteSize, infoBytes);
    //
    ////create descriptor set and update all need things
    //VkDescriptorSet descSet;
    //std::vector<VkDescriptorSetLayout> sets;
    //sets.push_back(computeDescriptorSetLayout);
    //VkUtil::createDescriptorSets(device, sets, descriptorPool, &descSet);
    //VkUtil::updateDescriptorSet(device, infos, infosByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    //
    //std::vector<VkImageLayout> imageLayouts(required3dImages, VK_IMAGE_LAYOUT_GENERAL);
    //VkUtil::updateStorageImageArrayDescriptorSet(device, image3dSampler, image3dView, imageLayouts, 1, descSet);
    //VkUtil::updateDescriptorSet(device, indices, amtOfIndices * sizeof(uint32_t), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    //VkUtil::updateDescriptorSet(device, data, amtOfData * sizeof(float), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    //
    ////creating the command buffer, binding all the needed things and dispatching it to update the density images
    //VkCommandBuffer computeCommands;
    //VkUtil::createCommandBuffer(device, commandPool, &computeCommands);
    //
    //vkCmdBindPipeline(computeCommands, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    //vkCmdBindDescriptorSets(computeCommands, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, { 0 });
    //uint32_t patchAmount = amtOfIndices / LOCALSIZE;
    //patchAmount += (amtOfIndices % LOCALSIZE) ? 1 : 0;
    //vkCmdDispatch(computeCommands, patchAmount, 1, 1);
    //VkUtil::commitCommandBuffer(queue, computeCommands);
    //err = vkQueueWaitIdle(queue);
    //check_vk_result(err);
    //
    //vkFreeCommandBuffers(device, commandPool, 1, &computeCommands);
    //vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
    //vkFreeMemory(device, infosMem, nullptr);
    //vkDestroyBuffer(device, infos, nullptr);
    //delete[] infoBytes;

    //uploading the density values currently manual, as there is an error in the compute pipeline ----------------------------------------------------------

    float** densityImages = new float*[required3dImages];
    for (int i = 0; i < required3dImages; ++i) {
        densityImages[i] = new float[w * d * h];
        for (int j = 0; j < w * d * h; ++j) {
            densityImages[i][j] = std::numeric_limits<float>::infinity();
        }
    }
    bool error = false;
    for (int i : indices) {
        int x = SpacialData::getRlatIndex(data[i][positionIndices.x]);
        int y = SpacialData::getAltitudeIndex(data[i][positionIndices.y]);
        if (y > h - 44)
            y = (y - h + 44) * 2 + (h - 44);
        int z = SpacialData::getRlonIndex(data[i][positionIndices.z]);
        if (x < 0 || y < 0 || z < 0) { 
            error = true; 
            break;
        }

        for (int j = 0; j < required3dImages; ++j) {
            densityImages[j][IDX3D(x, y, z, w, h)] = data[i][densityAttributes[j]];
            if (y >= h - 44) densityImages[j][IDX3D(x, y + 1, z, w, h)] = data[i][densityAttributes[j]];
        }
    }
    
    for (int i = 0; i < required3dImages; ++i) {
        if(!error)
            VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, image3d[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_FORMAT_R32_SFLOAT, w, h, d, densityImages[i], w * h * d * sizeof(float));
        delete[] densityImages[i];
    }
    delete[] densityImages;
    if (error) return false;

    VkUtil::createCommandBuffer(device, commandPool, &imageCommands);
    for (int i = 0; i < required3dImages; ++i) {
        VkUtil::transitionImageLayout(imageCommands, image3d[i], VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    VkUtil::commitCommandBuffer(queue, imageCommands);
    err = vkQueueWaitIdle(queue);
    check_vk_result(err);
    vkFreeCommandBuffers(device, commandPool, 1, &imageCommands);

    //end of uploading the density values ------------------------------------------------------------------------------------------------------------------

    //setting up all resources for the compute pipeline to create the binary volume

    if (index == -1) {
        binaryImage.push_back({});
        binarySmooth.push_back({});
        binaryImageView.push_back({});
        binarySmoothView.push_back({});
        binaryImageMemory.push_back({});
        VkUtil::create3dImage(device, w, h, d, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &binaryImage.back());
        VkUtil::create3dImage(device, width, height, depth, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &binarySmooth.back());
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, binaryImage.back(), &memReq);
        vkGetImageMemoryRequirements(device, binarySmooth.back(), &memReq);
        memReq.size *= 2;
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
        check_vk_result(vkAllocateMemory(device, &allocInfo, nullptr, &binaryImageMemory.back()));
        vkBindImageMemory(device, binaryImage.back(), binaryImageMemory.back(), 0);
        vkBindImageMemory(device, binarySmooth.back(), binaryImageMemory.back(), memReq.size / 2);

        VkUtil::create3dImageView(device, binaryImage.back(), VK_FORMAT_R8_UNORM, 1, &binaryImageView.back());
        VkUtil::create3dImageView(device, binarySmooth.back(), VK_FORMAT_R8_UNORM, 1, &binarySmoothView.back());
    }

    VkCommandBuffer binaryCommands;
    VkUtil::createCommandBuffer(device, commandPool, &binaryCommands);

    uint32_t binaryComputeInfosSize = sizeof(BinaryComputeInfos) + 2 * amtOfAttributes * sizeof(float);
    for (auto axis : brush) {
        binaryComputeInfosSize += axis.size() * 2 * sizeof(float);
    }
    BinaryComputeInfos* binaryComputeInfos = (BinaryComputeInfos*)new char[binaryComputeInfosSize];
    binaryComputeInfos->amtOfAxis = amtOfAttributes;
    binaryComputeInfos->maxX = w;
    binaryComputeInfos->maxY = h;
    binaryComputeInfos->maxZ = d;
    float* brushI = (float*)(binaryComputeInfos + 1);

    uint32_t curOffset = amtOfAttributes;        //the first offset is for axis 1, which is the size of the axis
    for (int axis = 0; axis < amtOfAttributes; ++axis) {
        brushI[axis] = curOffset;
        brushI[curOffset++] = brush[axis].size();
        for (int b = 0; b < brush[axis].size(); ++b) {
            brushI[curOffset++] = brush[axis][b].first;
            brushI[curOffset++] = brush[axis][b].second;
        }
    }

#ifdef _DEBUG
    assert(sizeof(BinaryComputeInfos) + curOffset * sizeof(float) == binaryComputeInfosSize);

    //PCUtil::numdump(brushI, curOffset);

    //testing the brush infos
    //for (int axis = 0; axis < binaryComputeInfos->amtOfAxis; ++axis) {
    //    int axisOffset = int(brushI[axis]);
    //    //check if there exists a brush on this axis
    //    if (bool(brushI[axisOffset])) {        //amtOfBrushes > 0
    //        //as there exist brushes we get the density for this attribute
    //        //float density = imageLoad(densities[axis], ivec3(gl_GlobalInvocationID)).x;
    //        bool inside = true;
    //        //for every brush
    //        for (int brush = 0; brush < brushI[axisOffset]; ++brush) {
    //            //for every MinMax
    //            int minMaxOffset = axisOffset + 1 + 2 * brush;            //+6 as after 1 the brush index lies, then the amtount of Minmax lies and then the color comes in a vec4
    //            //int brushIndex = int(info.brushes[brushOffset]);
    //            float mi = brushI[minMaxOffset];
    //            float ma = brushI[minMaxOffset + 1];
    //            //if (density<mi || density>ma) {
    //            //    inside = false;
    //            //    break;
    //            //}
    //        }
    //        if (!inside) {            //write 0 into binary texture and early out
    //            //imageStore(binary, ivec3(gl_GlobalInvocationID), vec4(0));
    //            return;
    //        }
    //    }
    //}
#endif

    VkBuffer binaryComputeBuffer;
    VkDeviceMemory binaryComputeMemory;
    VkUtil::createBuffer(device, binaryComputeInfosSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &binaryComputeBuffer);
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, binaryComputeBuffer, &memReq);
    allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &binaryComputeMemory);
    vkBindBufferMemory(device, binaryComputeBuffer, binaryComputeMemory, 0);
    VkUtil::uploadData(device, binaryComputeMemory, 0, binaryComputeInfosSize, binaryComputeInfos);

    VkDescriptorSet binaryDescriptorSet;
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(binaryComputeDescriptorSetLayout);
    VkUtil::createDescriptorSets(device, layouts, descriptorPool, &binaryDescriptorSet);

    VkUtil::updateDescriptorSet(device, binaryComputeBuffer, binaryComputeInfosSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binaryDescriptorSet);
    std::vector<VkImageLayout> imageLayouts(required3dImages, VK_IMAGE_LAYOUT_GENERAL);
    VkUtil::updateStorageImageArrayDescriptorSet(device, image3dSampler, image3dView, imageLayouts, 1, binaryDescriptorSet);
    VkUtil::updateStorageImageDescriptorSet(device, (index == -1)?binaryImageView.back() : binaryImageView[index], VK_IMAGE_LAYOUT_GENERAL, 2, binaryDescriptorSet);

    VkUtil::transitionImageLayout(binaryCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(binaryCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    VkUtil::transitionImageLayout(imageCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

    vkCmdBindPipeline(binaryCommands, VK_PIPELINE_BIND_POINT_COMPUTE, binaryComputePipeline);
    vkCmdBindDescriptorSets(binaryCommands, VK_PIPELINE_BIND_POINT_COMPUTE, binaryComputePipelineLayout, 0, 1, &binaryDescriptorSet, 0, nullptr);
    uint32_t patchAmtX = w / LOCALSIZE3D + ((w % LOCALSIZE) ? 1 : 0);
    uint32_t patchAmtY = h / LOCALSIZE3D + ((h % LOCALSIZE) ? 1 : 0);
    uint32_t patchAmtZ = d / LOCALSIZE3D + ((d % LOCALSIZE) ? 1 : 0);
    vkCmdDispatch(binaryCommands, patchAmtX, patchAmtY, patchAmtZ);
    //VkUtil::transitionImageLayout(binaryCommands, binaryImage, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkUtil::commitCommandBuffer(queue, binaryCommands);
    check_vk_result(vkQueueWaitIdle(queue));

    vkDestroyBuffer(device, binaryComputeBuffer, nullptr);
    vkFreeMemory(device, binaryComputeMemory, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &binaryCommands);
    delete[] binaryComputeInfos;
    //checking binary values
    //unsigned char* bi = new unsigned char[w * h * d];
    //int nonzero = 0;
    //VkUtil::downloadImageData(device, physicalDevice, commandPool, queue, binaryImage, w, h, d, bi, w * h * d);
    //for (int i = 0; i < width * depth * height; ++i) {
    //    if (bi[i] == 0) {
    //        nonzero++;
    //    }
    //}
    //std::cout << "Number of zero values: " << nonzero<< std::endl;
    //delete[] bi;

    if (!descriptorSet) {
        resize(1, 1);
        return true;
    }

    smoothImage((index == -1) ? binaryImage.size() - 1 : index);
    updateBrushBuffer();
    updateDescriptorSet();
    updateCommandBuffer();
    render();
    return true;
}

bool IsoSurfRenderer::update3dBinaryVolume(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim, uint32_t amtOfAttributes, const std::vector<uint32_t>& brushAttributes, const std::vector<std::pair<float, float>>& densityAttributesMinMax, glm::uvec3& positionIndices, VkBuffer data, uint32_t dataByteSize, VkBuffer indices, uint32_t amtOfIndices, std::vector<std::vector<std::pair<float, float>>>& brush, int index)
{
    if(binaryImage.size() && index != -1 && (drawlistBrushes[index].gridDimensions[0] != xDim.size() || drawlistBrushes[index].gridDimensions[1] != yDim.size() || drawlistBrushes[index].gridDimensions[2] != zDim.size())) return false;
    int width = xDim.size(), height = yDim.size(), depth = zDim.size();

    VkResult err;

    if (!binaryImageSampler) {
        VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 1, 1, &binaryImageSampler);
    }

    updateDimensionImages(xDim, yDim, zDim);

    if (index == -1)
        posIndices.push_back(positionIndices);
    else {
        positionIndices = posIndices[index];
    }

    VkClearColorValue clear = { 0,0,0,0 };
    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1 };

    //creating the binary image via compute pipeline ----------------------------------------
    if (index == -1) {
        binaryImage.push_back({});
        binarySmooth.push_back({});
        binaryImageView.push_back({});
        binarySmoothView.push_back({});
        binaryImageMemory.push_back({});
        VkUtil::create3dImage(device, width, height, depth, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &binaryImage.back());
        VkUtil::create3dImage(device, width, height, depth, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &binarySmooth.back());
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, binaryImage.back(), &memReq);
        vkGetImageMemoryRequirements(device, binarySmooth.back(), &memReq);
        memReq.size *= 2;
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
        check_vk_result(vkAllocateMemory(device, &allocInfo, nullptr, &binaryImageMemory.back()));
        vkBindImageMemory(device, binaryImage.back(), binaryImageMemory.back(), 0);
        vkBindImageMemory(device, binarySmooth.back(), binaryImageMemory.back(), memReq.size / 2);

        VkUtil::create3dImageView(device, binaryImage.back(), VK_FORMAT_R8_UNORM, 1, &binaryImageView.back());
        VkUtil::create3dImageView(device, binarySmooth.back(), VK_FORMAT_R8_UNORM, 1, &binarySmoothView.back());
    }

    VkBuffer infos;
    VkDeviceMemory infosMem;
    uint32_t infosByteSize = sizeof(ComputeInfos) + brushAttributes.size() * sizeof(float) * 3;
    ComputeInfos* infoBytes = (ComputeInfos*)new char[infosByteSize];
    VkUtil::createBuffer(device, infosByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &infos);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq = {};
    vkGetBufferMemoryRequirements(device, infos, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &infosMem);
    vkBindBufferMemory(device, infos, infosMem, 0);
    
    //fill infoBytes and upload it
    infoBytes->amtOfAttributes = amtOfAttributes;
    infoBytes->amtOfBrushAttributes = brushAttributes.size();
    infoBytes->amtOfIndices = amtOfIndices;
    infoBytes->dimX = width;
    infoBytes->dimY = height;
    infoBytes->dimZ = depth;
    infoBytes->xInd = positionIndices.x;
    infoBytes->yInd = positionIndices.y;
    infoBytes->zInd = positionIndices.z;
    infoBytes->xMin = densityAttributesMinMax[positionIndices.x].first;
    infoBytes->xMax = densityAttributesMinMax[positionIndices.x].second;
    infoBytes->yMin = densityAttributesMinMax[positionIndices.y].first;
    infoBytes->yMax = densityAttributesMinMax[positionIndices.y].second;
    infoBytes->zMin = densityAttributesMinMax[positionIndices.z].first;
    infoBytes->zMax = densityAttributesMinMax[positionIndices.z].second;
    infoBytes->regularGrid = (uint32_t(dimensionCorrectionLinearDim[0])) | (uint32_t(dimensionCorrectionLinearDim[1]) << 1) | (uint32_t(dimensionCorrectionLinearDim[2]) << 2);
    int* inf = (int*)(infoBytes + 1);
    int offset = 0;
    for (int i = 0; i < brushAttributes.size(); ++i) {
        inf[i * 3] = brushAttributes[i];
        inf[i * 3 + 1] = brush[brushAttributes[i]].size();
        inf[i * 3 + 2] = offset;
        offset += brush[brushAttributes[i]].size() * 2;
    }
    PCUtil::numdump(inf, brushAttributes.size() * 3);
    //assert(infosByteSize == offset * sizeof(float) + sizeof(ComputeInfos));
    VkUtil::uploadData(device, infosMem, 0, infosByteSize, infoBytes);

    //create graphics buffer for brushes
    uint32_t brushByteSize = offset * sizeof(float);        //the offset has already counted the amount of brushes that there are going to be
    float* brushBytes = new float[offset];
    VkBuffer brushBuffer;
    VkDeviceMemory brushMemory;
    VkUtil::createBuffer(device, brushByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &brushBuffer);
    vkGetBufferMemoryRequirements(device, brushBuffer, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &brushMemory);
    vkBindBufferMemory(device, brushBuffer, brushMemory, 0);
    offset = 0;
    for (int i = 0; i < brush.size(); ++i) {
        for (auto& minMax : brush[i]) {
            brushBytes[offset++] = minMax.first;
            brushBytes[offset++] = minMax.second;
        }
    }
    //PCUtil::numdump(brushBytes, offset);
    assert(offset * sizeof(float) == brushByteSize);
    VkUtil::uploadData(device, brushMemory, 0, brushByteSize, brushBytes);
    delete[] brushBytes;

    //create graphics buffer for dimension values
    uint32_t dimValsByteSize = (4 + xDim.size() + yDim.size() + zDim.size()) * sizeof(float);
    float* dimValsBytes = new float[dimValsByteSize];
    VkBuffer dimValsBuffer;
    VkDeviceMemory dimValsMemory;
    VkUtil::createBuffer(device, dimValsByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dimValsBuffer);
    vkGetBufferMemoryRequirements(device, dimValsBuffer, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &dimValsMemory);
    vkBindBufferMemory(device, dimValsBuffer, dimValsMemory, 0);
    offset = 0;
    dimValsBytes[offset++] = xDim.size();
    dimValsBytes[offset++] = yDim.size();
    dimValsBytes[offset++] = zDim.size();
    dimValsBytes[offset++] = 0; //padding
    for (float f : xDim) {
        dimValsBytes[offset++] = f;
    }
    for (float f : yDim) {
        dimValsBytes[offset++] = f;
    }
    for (float f : zDim) {
        dimValsBytes[offset++] = f;
    }
    assert(offset * sizeof(float) == dimValsByteSize);
    VkUtil::uploadData(device, dimValsMemory, 0, dimValsByteSize, dimValsBytes);
    delete[] dimValsBytes;

    //create descriptor set and update all need things
    VkDescriptorSet descSet;
    std::vector<VkDescriptorSetLayout> sets;
    sets.push_back(computeDescriptorSetLayout);
    VkUtil::createDescriptorSets(device, sets, descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(device, infos, infosByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, brushBuffer, brushByteSize, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, dimValsBuffer, dimValsByteSize, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, indices, amtOfIndices * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, data, VK_WHOLE_SIZE, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateStorageImageDescriptorSet(device, (index == -1) ? binaryImageView.back() : binaryImageView[index], VK_IMAGE_LAYOUT_GENERAL, 5, descSet);
    
    //creating the command buffer, binding all the needed things and dispatching it to update the density images
    VkCommandBuffer computeCommands;
    VkUtil::createCommandBuffer(device, commandPool, &computeCommands);
    
    vkCmdBindPipeline(computeCommands, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    if(index == -1)
        VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    else
        VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindDescriptorSets(computeCommands, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, { 0 });
    uint32_t patchAmount = amtOfIndices / LOCALSIZE;
    patchAmount += (amtOfIndices % LOCALSIZE) ? 1 : 0;
    vkCmdDispatch(computeCommands, patchAmount, 1, 1);
    VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkUtil::commitCommandBuffer(queue, computeCommands);
    err = vkQueueWaitIdle(queue);
    check_vk_result(err);
    
    vkDestroyBuffer(device, brushBuffer, nullptr);
    vkFreeMemory(device, brushMemory, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &computeCommands);
    vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
    vkFreeMemory(device, infosMem, nullptr);
    vkDestroyBuffer(device, infos, nullptr);
    vkFreeMemory(device, dimValsMemory, nullptr);
    vkDestroyBuffer(device, dimValsBuffer, nullptr);
    delete[] infoBytes;

    if (!descriptorSet) {
        resize(1, 1);
        return true;
    }

    smoothImage((index == -1)?binaryImage.size() - 1:index);
    updateBrushBuffer();
    updateDescriptorSet();
    updateCommandBuffer();
    render();
    return true;
}

IsoSurfRenderer::IsoSurfRendererError IsoSurfRenderer::update3dBinaryVolume(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim, uint32_t posIndices[3], std::vector<std::pair<float, float>>& posBounds, uint32_t amtOfAttributes, uint32_t dataSize, VkBuffer data, VkBufferView activeIndices, uint32_t indicesSize, VkBuffer indices, bool regularGrid[3], int index)
{
    int width = xDim.size(), height = yDim.size(), depth = zDim.size();
    if (index != -1 && (drawlistBrushes[index].gridDimensions[0]!= width || drawlistBrushes[index].gridDimensions[1] != height || drawlistBrushes[index].gridDimensions[2] != depth)) return IsoSurfRendererError_GridDimensionMissmatch;
    
    dimensionCorrectionLinearDim[0] = regularGrid[0];
    dimensionCorrectionLinearDim[1] = regularGrid[1];
    dimensionCorrectionLinearDim[2] = regularGrid[2];
    updateDimensionImages(xDim, yDim, zDim);

    VkResult err;

    if (!binaryImageSampler) {
        VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 1, 1, &binaryImageSampler);
    }

    if (index == -1)
        this->posIndices.push_back({ posIndices[0],posIndices[1],posIndices[2] });
    else {
        posIndices[0] = this->posIndices[index][0];
        posIndices[1] = this->posIndices[index][1];
        posIndices[2] = this->posIndices[index][2];
    }

    VkClearColorValue clear = { 0,0,0,0 };
    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1 };

    //creating the binary image via compute pipeline ----------------------------------------
    if (index == -1) {
        binaryImage.push_back({});
        binarySmooth.push_back({});
        binaryImageView.push_back({});
        binarySmoothView.push_back({});
        binaryImageMemory.push_back({});
        VkUtil::create3dImage(device, width, height, depth, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &binaryImage.back());
        VkUtil::create3dImage(device, width, height, depth, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &binarySmooth.back());
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, binaryImage.back(), &memReq);
        vkGetImageMemoryRequirements(device, binarySmooth.back(), &memReq);
        memReq.size *= 2;
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
        check_vk_result(vkAllocateMemory(device, &allocInfo, nullptr, &binaryImageMemory.back()));
        vkBindImageMemory(device, binaryImage.back(), binaryImageMemory.back(), 0);
        vkBindImageMemory(device, binarySmooth.back(), binaryImageMemory.back(), memReq.size / 2);

        VkUtil::create3dImageView(device, binaryImage.back(), VK_FORMAT_R8_UNORM, 1, &binaryImageView.back());
        VkUtil::create3dImageView(device, binarySmooth.back(), VK_FORMAT_R8_UNORM, 1, &binarySmoothView.back());
    }

    VkBuffer infos;
    VkDeviceMemory infosMem;
    uint32_t infosByteSize = sizeof(ComputeInfos);
    ComputeInfos* infoBytes = (ComputeInfos*)new char[infosByteSize];
    VkUtil::createBuffer(device, infosByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &infos);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReq = {};
    vkGetBufferMemoryRequirements(device, infos, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &infosMem);
    vkBindBufferMemory(device, infos, infosMem, 0);

    //fill infoBytes and upload it
    infoBytes->amtOfAttributes = amtOfAttributes;
    infoBytes->amtOfBrushAttributes = 0;
    infoBytes->amtOfIndices = indicesSize;
    infoBytes->dimX = width;
    infoBytes->dimY = height;
    infoBytes->dimZ = depth;
    infoBytes->xInd = posIndices[0];
    infoBytes->yInd = posIndices[1];
    infoBytes->zInd = posIndices[2];
    infoBytes->xMin = posBounds[0].first;
    infoBytes->xMax = posBounds[0].second;
    infoBytes->yMin = posBounds[1].first;
    infoBytes->yMax = posBounds[1].second;
    infoBytes->zMin = posBounds[2].first;
    infoBytes->zMax = posBounds[2].second;
    infoBytes->regularGrid = (uint32_t(dimensionCorrectionLinearDim[0])) | (uint32_t(dimensionCorrectionLinearDim[1]) << 1) | (uint32_t(dimensionCorrectionLinearDim[2]) << 2);
    VkUtil::uploadData(device, infosMem, 0, infosByteSize, infoBytes);

    //create graphics buffer for dimension values
    uint32_t dimValsByteSize = (4 + xDim.size() + yDim.size() + zDim.size()) * sizeof(float);
    float* dimValsBytes = new float[dimValsByteSize];
    VkBuffer dimValsBuffer;
    VkDeviceMemory dimValsMemory;
    VkUtil::createBuffer(device, dimValsByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dimValsBuffer);
    vkGetBufferMemoryRequirements(device, dimValsBuffer, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &dimValsMemory);
    vkBindBufferMemory(device, dimValsBuffer, dimValsMemory, 0);
    int offset = 0;
    dimValsBytes[offset++] = xDim.size();
    dimValsBytes[offset++] = yDim.size();
    dimValsBytes[offset++] = zDim.size();
    dimValsBytes[offset++] = 0; //padding
    for (float f : xDim) {
        dimValsBytes[offset++] = f;
    }
    for (float f : yDim) {
        dimValsBytes[offset++] = f;
    }
    for (float f : zDim) {
        dimValsBytes[offset++] = f;
    }
    assert(offset * sizeof(float) == dimValsByteSize);
    VkUtil::uploadData(device, dimValsMemory, 0, dimValsByteSize, dimValsBytes);
    delete[] dimValsBytes;

    //create descriptor set and update all need things
    VkDescriptorSet descSet;
    std::vector<VkDescriptorSetLayout> sets;
    sets.push_back(activeIndComputeDescriptorSetLayout);
    VkUtil::createDescriptorSets(device, sets, descriptorPool, &descSet);
    VkUtil::updateDescriptorSet(device, infos, infosByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateTexelBufferDescriptorSet(device, activeIndices, 1, descSet);
    VkUtil::updateDescriptorSet(device, indices, indicesSize * sizeof(uint32_t), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, data, VK_WHOLE_SIZE, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateDescriptorSet(device, dimValsBuffer, dimValsByteSize, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
    VkUtil::updateStorageImageDescriptorSet(device, (index == -1) ? binaryImageView.back() : binaryImageView[index], VK_IMAGE_LAYOUT_GENERAL, 5, descSet);

    //creating the command buffer, binding all the needed things and dispatching it to update the density images
    VkCommandBuffer computeCommands;
    VkUtil::createCommandBuffer(device, commandPool, &computeCommands);

    vkCmdBindPipeline(computeCommands, VK_PIPELINE_BIND_POINT_COMPUTE, activeIndComputePipeline);
    if (index == -1)
        VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    else
        VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindDescriptorSets(computeCommands, VK_PIPELINE_BIND_POINT_COMPUTE, activeIndComputePipelineLayout, 0, 1, &descSet, 0, { 0 });
    uint32_t patchAmount = indicesSize / LOCALSIZE;
    patchAmount += (indicesSize % LOCALSIZE) ? 1 : 0;
    vkCmdDispatch(computeCommands, patchAmount, 1, 1);
    VkUtil::transitionImageLayout(computeCommands, (index == -1) ? binaryImage.back() : binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkUtil::commitCommandBuffer(queue, computeCommands);
    err = vkQueueWaitIdle(queue);
    check_vk_result(err);

    vkFreeCommandBuffers(device, commandPool, 1, &computeCommands);
    vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
    vkFreeMemory(device, infosMem, nullptr);
    vkDestroyBuffer(device, infos, nullptr);
    vkFreeMemory(device, dimValsMemory, nullptr);
    vkDestroyBuffer(device, dimValsBuffer, nullptr);
    delete[] infoBytes;

    if (!descriptorSet) {
        resize(1, 1);
        return IsoSurfRendererError_Success;
    }

    smoothImage((index == -1) ? binaryImage.size() - 1 : index);
    updateBrushBuffer();
    updateDescriptorSet();
    updateCommandBuffer();
    render();
    return IsoSurfRendererError_Success;
}

void IsoSurfRenderer::deleteBinaryVolume(uint32_t ind)
{
    vkDestroyImage(device, binaryImage[ind], nullptr);
    vkDestroyImageView(device, binaryImageView[ind], nullptr);
    vkDestroyImage(device, binarySmooth[ind], nullptr);
    vkDestroyImageView(device, binarySmoothView[ind], nullptr);
    vkFreeMemory(device, binaryImageMemory[ind], nullptr);
    for (int i = ind; i < drawlistBrushes.size() - 1; ++i) {
        drawlistBrushes[i] = drawlistBrushes[i + 1];
        posIndices[i] = posIndices[i + 1];
        binaryImage[i] = binaryImage[i + 1];
        binaryImageView[i] = binaryImageView[i + 1];
        binarySmooth[i] = binarySmooth[i + 1];
        binarySmoothView[i] = binarySmoothView[i + 1];
        binaryImageMemory[i] = binaryImageMemory[i + 1];
    }
    drawlistBrushes.pop_back();
    posIndices.pop_back();
    binaryImage.pop_back();
    binaryImageView.pop_back();
    binarySmooth.pop_back();
    binarySmoothView.pop_back();
    binaryImageMemory.pop_back();

    if(drawlistBrushes.size())
        updateDescriptorSet();
    updateCommandBuffer();
}

void IsoSurfRenderer::getPosIndices(int index, uint32_t* ind)
{
    ind[0] = posIndices[index].x;
    ind[1] = posIndices[index].y;
    ind[2] = posIndices[index].z;
}

void IsoSurfRenderer::updateCameraPos(CamNav::NavigationInput input, float deltaT)
{
    //first do the rotation, as the user has a more inert feeling when the fly direction matches the view direction instantly
    if (input.mouseDeltaX) {
        cameraRot.y -= rotationSpeed * input.mouseDeltaX * deltaT;
    }
    if (input.mouseDeltaY) {
        cameraRot.x -= rotationSpeed * input.mouseDeltaY * deltaT;
    }

    glm::mat4 rot = glm::eulerAngleYX(cameraRot.y, cameraRot.x);
    if (input.a) {    //fly left
        glm::vec4 left = rot * glm::vec4(-1, 0, 0, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
        cameraPos += glm::vec3(left.x, left.y, left.z);
    }
    if (input.d) {    //fly right
        glm::vec4 right = rot * glm::vec4(1, 0, 0, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
        cameraPos += glm::vec3(right.x, right.y, right.z);
    }
    if (input.s) {    //fly backward
        glm::vec4 back = rot * glm::vec4(0, 0, 1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
        cameraPos += glm::vec3(back.x, back.y, back.z);
    }
    if (input.w) {    //fly forward
        glm::vec4 front = rot * glm::vec4(0, 0, -1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
        cameraPos += glm::vec3(front.x, front.y, front.z);
    }
    if (input.q) {    //fly down
        cameraPos += glm::vec3(0, -1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
    }
    if (input.e) {    //fly up
        cameraPos += glm::vec3(0, 1, 0) * flySpeed * ((input.shift) ? fastFlyMultiplier : 1) * deltaT;
    }
}


void IsoSurfRenderer::setCameraPos(glm::vec3& newCameraPos, float** newRotation) {
    cameraPos = newCameraPos;
    cameraRot.x = (*newRotation)[0];
    cameraRot.y = (*newRotation)[1];
    cameraRotationGUI[0] = (*newRotation)[0];
    cameraRotationGUI[1] = (*newRotation)[1];
    return;
}


void IsoSurfRenderer::getCameraPos(glm::vec3& cameraPosReturn, float** rotationReturn) {
    cameraPosReturn = cameraPos;
    (*rotationReturn)[0] = cameraRot.x;
    (*rotationReturn)[1] = cameraRot.y;
    return;
}


void IsoSurfRenderer::addBrush(std::string& name, std::vector<std::vector<std::pair<float, float>>> minMax)
{
    updateBrushBuffer();
    updateDescriptorSet();
    updateCommandBuffer();
    render();
}

bool IsoSurfRenderer::updateBrush(std::string& name, std::vector<std::vector<std::pair<float, float>>> minMax)
{
    if (brushes.find(name) == brushes.end()) return false;

    brushes[name] = minMax;
    
    updateBrushBuffer();
    updateDescriptorSet();
    render();

    return true;
}

bool IsoSurfRenderer::deleteBrush(std::string& name)
{
    return brushes.erase(name) > 0;
}

void IsoSurfRenderer::render()
{
    assert(drawlistBrushes.size() == binaryImage.size());

    if (!drawlistBrushes.size())
        return;

    VkResult err;

    //uploading the uniformBuffer
    UniformBuffer ubo;
    ubo.mvp = glm::perspective(glm::radians(45.0f), (float)imageWidth / (float)imageHeight, 0.1f, 100.0f);;
    ubo.mvp[1][1] *= -1;
    glm::mat4 view = glm::transpose(glm::eulerAngleY(cameraRot.y) * glm::eulerAngleX(cameraRot.x)) * glm::translate(glm::mat4(1.0), -cameraPos);;
    glm::mat4 scale = glm::scale(glm::mat4(1.0f),glm::vec3(boxWidth,boxHeight,boxDepth));
    ubo.mvp = ubo.mvp * view *scale;
    ubo.camPos = glm::inverse(scale) * glm::vec4(cameraPos, gridLineWidth);

    ubo.faces.x = float(ubo.camPos.x > 0) - .5f;
    ubo.faces.y = float(ubo.camPos.y > 0) - .5f;
    ubo.faces.z = float(ubo.camPos.z > 0) - .5f;

    ubo.lightDir = lightDir;
    void* d;
    vkMapMemory(device, constantMemory, uniformBufferOffset, sizeof(UniformBuffer), 0, &d);
    memcpy(d, &ubo, sizeof(UniformBuffer));
    vkUnmapMemory(device, constantMemory);

    uint32_t brushInfosSize = sizeof(BrushInfos) + 4 * sizeof(float) * drawlistBrushes.size();
    BrushInfos* brushInfos = (BrushInfos*)new char[brushInfosSize];
    brushInfos->amtOfAxis = binaryImage.size();
    brushInfos->shade = shade;
    brushInfos->stepSize = stepSize;
    brushInfos->isoValue = isoValue;
    brushInfos->shadingStep = shadingStep;
    brushInfos->linearDims = (uint32_t(dimensionCorrectionLinearDim[0])) | (uint32_t(dimensionCorrectionLinearDim[1]) << 1) | (uint32_t(dimensionCorrectionLinearDim[2]) << 2);
    float* brushColors = (float*)(brushInfos + 1);
    for (int i = 0; i < drawlistBrushes.size(); ++i) {
        brushColors[i * 4] = drawlistBrushes[i].brushSurfaceColor.x;
        brushColors[i * 4 + 1] = drawlistBrushes[i].brushSurfaceColor.y;
        brushColors[i * 4 + 2] = drawlistBrushes[i].brushSurfaceColor.z;
        brushColors[i * 4 + 3] = drawlistBrushes[i].brushSurfaceColor.w;
    }
    VkUtil::uploadData(device, brushMemory, 0, brushInfosSize, brushInfos);

    delete[] brushInfos;

    //submitting the command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    check_vk_result(err);
}

void IsoSurfRenderer::setImageDescriptorSet(VkDescriptorSet descriptor)
{
    imageDescriptorSet = descriptor;
}

VkDescriptorSet IsoSurfRenderer::getImageDescriptorSet()
{
    return imageDescriptorSet;
}

VkSampler IsoSurfRenderer::getImageSampler()
{
    return sampler;
}

VkImageView IsoSurfRenderer::getImageView()
{
    return imageView;
}

void IsoSurfRenderer::exportBinaryCsv(std::string path, uint32_t binaryIndex)
{
    if (binaryIndex >= drawlistBrushes.size()) return;

    uint32_t w = drawlistBrushes[binaryIndex].gridDimensions[0];
    uint32_t h = drawlistBrushes[binaryIndex].gridDimensions[1];
    uint32_t d = drawlistBrushes[binaryIndex].gridDimensions[2];
    uint8_t* binaryData = new uint8_t[w * h * d];
    VkUtil::downloadImageData(device, physicalDevice, commandPool, queue, binaryImage[binaryIndex], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, w, h, d, binaryData, w * h * d); 

    if (true) {
        std::ofstream file(path);

        //int count = 0;
        for (int i = 0; i < w * h * d; ++i) {
            //if (binaryData[i]) ++count;
            // lat alt lon
            // alt, lat, lon

            // Order first all 500, then - 57- 700
    //        int alt = (i / w) % h;
    //        int lat = (i%w);
    //        int lon = (i / (w*h));

    //        file << alt /57.0 << "," << lat / 500.  << "," << lon / 700. << ",";
            file << std::to_string(binaryData[i]);
            //        if (i < w * h * d - 1) file << "\n";// ",";
            if (i < w * h * d - 1) file << ",";






        }
        //std::cout << count << std::endl;

        file.close();
    }

    if (false) {
        std::ofstream file(path);
        for (int i = 0; i < w * h * d; ++i) {
            file.write((char*)&binaryData[i], sizeof(binaryData[i]));
            std::cout << (char*)&binaryData[i];
        }
    }


    delete[] binaryData;
}

void IsoSurfRenderer::setBinarySmoothing(float stdDiv, bool keepOnes)
{
    smoothStdDiv = stdDiv;
    this->keepOnes = keepOnes;
    for (int i = 0; i < binaryImage.size(); ++i) {
        smoothImage(i);
    }
}

void IsoSurfRenderer::imageBackGroundUpdated()
{
    updateCommandBuffer();
}

void IsoSurfRenderer::smoothImage(int index)
{
    VkResult err;

    VkImage tmpImage;
    VkImageView tmpImageView;
    VkDeviceMemory tmpMemory;
    VkUtil::create3dImage(device, drawlistBrushes[index].gridDimensions[0], drawlistBrushes[index].gridDimensions[1], drawlistBrushes[index].gridDimensions[2], VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT, &tmpImage);
    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, tmpImage, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
    err = vkAllocateMemory(device, &allocInfo, nullptr, &tmpMemory);
    check_vk_result(err);
    vkBindImageMemory(device, tmpImage, tmpMemory, 0);
    VkUtil::create3dImageView(device, tmpImage, VK_FORMAT_R8_UNORM, 1, &tmpImageView);

    VkBuffer ubos;
    VkDeviceMemory uboMemory;
    uint32_t uboSize = (sizeof(SmoothUBO) % uboAlignment) ? sizeof(SmoothUBO) + uboAlignment - sizeof(SmoothUBO) % uboAlignment : sizeof(SmoothUBO);
    VkUtil::createBuffer(device, uboSize * 3, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubos);
    vkGetBufferMemoryRequirements(device, ubos, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check_vk_result(vkAllocateMemory(device, &allocInfo, nullptr, &uboMemory));
    vkBindBufferMemory(device, ubos, uboMemory, 0);

    SmoothUBO ubo{};
    ubo.index = 0;
    ubo.stdDev = smoothStdDiv;
    VkUtil::uploadData(device, uboMemory, 0, sizeof(SmoothUBO), &ubo);
    ubo.index = 1;
    VkUtil::uploadData(device, uboMemory, uboSize, sizeof(SmoothUBO), &ubo);
    ubo.index = 2;
    VkUtil::uploadData(device, uboMemory, 2 * uboSize, sizeof(SmoothUBO), &ubo);

    VkDescriptorSet descSets[4];
    std::vector<VkDescriptorSetLayout> layouts(3, binarySmoothDescriptorSetLayout);
    layouts.push_back(binaryCopyOnesDescriptorSetLayout);
    VkUtil::createDescriptorSets(device, layouts, descriptorPool, descSets);
    VkUtil::updateDescriptorSet(device, ubos, sizeof(SmoothUBO), 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descSets[0]);
    VkUtil::updateStorageImageDescriptorSet(device, binaryImageView[index], VK_IMAGE_LAYOUT_GENERAL, 1, descSets[0]);
    VkUtil::updateStorageImageDescriptorSet(device, binarySmoothView[index], VK_IMAGE_LAYOUT_GENERAL, 2, descSets[0]);
    VkUtil::updateDescriptorSet(device, ubos, sizeof(SmoothUBO), 0, uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descSets[1]);
    VkUtil::updateStorageImageDescriptorSet(device, binarySmoothView[index], VK_IMAGE_LAYOUT_GENERAL, 1, descSets[1]);
    VkUtil::updateStorageImageDescriptorSet(device, tmpImageView, VK_IMAGE_LAYOUT_GENERAL, 2, descSets[1]);
    VkUtil::updateDescriptorSet(device, ubos, sizeof(SmoothUBO), 0, 2 * uboSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descSets[2]);
    VkUtil::updateStorageImageDescriptorSet(device, tmpImageView, VK_IMAGE_LAYOUT_GENERAL, 1, descSets[2]);
    VkUtil::updateStorageImageDescriptorSet(device, binarySmoothView[index], VK_IMAGE_LAYOUT_GENERAL, 2, descSets[2]);

    VkUtil::updateStorageImageDescriptorSet(device, binaryImageView[index], VK_IMAGE_LAYOUT_GENERAL, 0, descSets[3]);
    VkUtil::updateStorageImageDescriptorSet(device, binarySmoothView[index], VK_IMAGE_LAYOUT_GENERAL, 1, descSets[3]);

    uint32_t patchAmtX = drawlistBrushes[index].gridDimensions[0] / LOCALSIZE3D + ((drawlistBrushes[index].gridDimensions[0] % LOCALSIZE3D) ? 1 : 0);
    uint32_t patchAmtY = drawlistBrushes[index].gridDimensions[1] / LOCALSIZE3D + ((drawlistBrushes[index].gridDimensions[1] % LOCALSIZE3D) ? 1 : 0);
    uint32_t patchAmtZ = drawlistBrushes[index].gridDimensions[2] / LOCALSIZE3D + ((drawlistBrushes[index].gridDimensions[2] % LOCALSIZE3D) ? 1 : 0);

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(device, commandPool, &commands);
    VkUtil::transitionImageLayout(commands, binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    VkUtil::transitionImageLayout(commands, binarySmooth[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    VkUtil::transitionImageLayout(commands, tmpImage, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, binarySmoothPipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, binarySmoothPipelineLayout, 0, 1, descSets, 0, nullptr);
    vkCmdDispatch(commands, patchAmtX, patchAmtY, patchAmtZ);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, binarySmoothPipelineLayout, 0, 1, &descSets[1], 0, nullptr);
    vkCmdDispatch(commands, patchAmtX, patchAmtY, patchAmtZ);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, binarySmoothPipelineLayout, 0, 1, &descSets[2], 0, nullptr);
    vkCmdDispatch(commands, patchAmtX, patchAmtY, patchAmtZ);
    if (keepOnes) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, binaryCopyOnesPipeline);
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, binaryCopyOnesPipelineLayout, 0, 1, &descSets[3], 0, nullptr);
        vkCmdDispatch(commands, patchAmtX, patchAmtY, patchAmtZ);
    }
    VkUtil::transitionImageLayout(commands, binarySmooth[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkUtil::transitionImageLayout(commands, binaryImage[index], VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkUtil::commitCommandBuffer(queue, commands);
    check_vk_result(vkQueueWaitIdle(queue));

    vkFreeCommandBuffers(device, commandPool, 1, &commands);
    vkDestroyBuffer(device, ubos, nullptr);
    vkFreeMemory(device, uboMemory, nullptr);
    vkDestroyImage(device, tmpImage, nullptr);
    vkDestroyImageView(device, tmpImageView, nullptr);
    vkFreeMemory(device, tmpMemory, nullptr);
    vkFreeDescriptorSets(device, descriptorPool, 4, descSets);
}

void IsoSurfRenderer::createPrepareImageCommandBuffer()
{
    VkUtil::createCommandBuffer(device, commandPool, &prepareImageCommand);
    VkUtil::transitionImageLayout(prepareImageCommand, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkEndCommandBuffer(prepareImageCommand);
}

void IsoSurfRenderer::createImageResources()
{
    VkResult err;
    
    VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT, &image);

    VkMemoryRequirements memReq = {};
    vkGetImageMemoryRequirements(device, image, &memReq);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
    err = vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
    check_vk_result(err);

    vkBindImageMemory(device, image, imageMemory, 0);

    VkUtil::createImageView(device, image, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &imageView);
    std::vector<VkImageView> views;
    views.push_back(imageView);
    VkUtil::createFrameBuffer(device, renderPass, views, imageWidth, imageHeight, &frameBuffer);
    VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, 16, 1, &sampler);
}

void IsoSurfRenderer::createBuffer()
{
    constexpr uint32_t alignment = 0x40;
    
    VkUtil::createBuffer(device, 8 * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertexBuffer);
    VkUtil::createBuffer(device, 12 * 3 * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &indexBuffer);
    VkUtil::createBuffer(device, sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uniformBuffer);

    VkResult err;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, vertexBuffer, &memReq);

    uint32_t memoryTypeBits = memReq.memoryTypeBits;
    VkMemoryAllocateInfo memAlloc = {};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.allocationSize = (memAlloc.allocationSize + alignment - 1) / alignment * alignment;
    uint32_t indexBufferOffset = memReq.size;

    vkGetBufferMemoryRequirements(device, indexBuffer, &memReq);
    memAlloc.allocationSize += memReq.size;
    memAlloc.allocationSize = (memAlloc.allocationSize + alignment - 1) / alignment * alignment;
    memoryTypeBits |= memReq.memoryTypeBits;
    uniformBufferOffset = memAlloc.allocationSize;

    vkGetBufferMemoryRequirements(device, uniformBuffer, &memReq);
    memAlloc.allocationSize += memReq.size;
    memAlloc.allocationSize = (memAlloc.allocationSize + alignment - 1) / alignment * alignment;
    memoryTypeBits |= memReq.memoryTypeBits;

    memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    err = vkAllocateMemory(device, &memAlloc, nullptr, &constantMemory);
    check_vk_result(err);

    vkBindBufferMemory(device, vertexBuffer, constantMemory, 0);
    vkBindBufferMemory(device, indexBuffer, constantMemory, indexBufferOffset);
    vkBindBufferMemory(device, uniformBuffer, constantMemory, uniformBufferOffset);

    //creating the data for the buffers
    glm::vec3 vB[8];
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                vB[(x << 2) | (y << 1) | z] = glm::vec3(x - .5f, y - .5f, z - .5f);
            }
        }
    }

    uint16_t iB[12 * 3] = { 0,1,2, 1,3,2, 0,4,1, 1,4,5, 0,2,4, 2,6,4, 2,3,6, 3,7,6, 4,6,5, 5,6,7, 1,5,7, 1,7,3 };

    void* d;
    vkMapMemory(device, constantMemory, 0, sizeof(vB), 0, &d);
    memcpy(d, vB, sizeof(vB));
    vkUnmapMemory(device, constantMemory);
    vkMapMemory(device, constantMemory, indexBufferOffset, sizeof(iB), 0, &d);
    memcpy(d, iB, sizeof(iB));
    vkUnmapMemory(device, constantMemory);
}

void IsoSurfRenderer::createPipeline()
{
    VkShaderModule shaderModules[5] = {};
    //the vertex shader for the pipeline
    std::vector<char> vertexBytes = PCUtil::readByteFile(vertPath);
    shaderModules[0] = VkUtil::createShaderModule(device, vertexBytes);
    //the fragment shader for the pipeline
    std::vector<char> fragmentBytes = PCUtil::readByteFile(fragPath);
    shaderModules[4] = VkUtil::createShaderModule(device, fragmentBytes);


    //Description for the incoming vertex attributes
    VkVertexInputBindingDescription bindingDescripiton = {};        //describes how big the vertex data is and how to read the data
    bindingDescripiton.binding = 0;
    bindingDescripiton.stride = sizeof(glm::vec3);
    bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription = {};    //describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
    attributeDescription.binding = 0;
    attributeDescription.location = 0;
    attributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescription.offset = offsetof(glm::vec3, x);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

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
    rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
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

    //creating the descriptor set layout
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    uboLayoutBinding.descriptorCount = MAXAMTOF3DTEXTURES;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 2;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 3;
    uboLayoutBinding.descriptorCount = 3;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings.push_back(uboLayoutBinding);

    std::vector<bool> valid{ true,false,true, true };
    VkUtil::createDescriptorSetLayoutPartiallyBound(device, bindings, valid, &descriptorSetLayout);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    descriptorSetLayouts.push_back(descriptorSetLayout);

    VkUtil::createRenderPass(device, VkUtil::PASS_TYPE_COLOR_OFFLINE, &renderPass);

    VkUtil::createPipeline(device, &vertexInputInfo, imageWidth, imageHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);

    // creating the compute pipeline to fill the density images --------------------------------------------------
    VkShaderModule computeModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(computePath));

    bindings.clear();
    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorCount = 1;                        
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binding.binding = 0;                                //compute infos
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(binding);

    binding.binding = 1;                                //brushes
    bindings.push_back(binding);

    binding.binding = 2;                                //axisvalues
    bindings.push_back(binding);

    binding.binding = 3;                                //indices buffer
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(binding);

    binding.binding = 4;                                //data buffer
    bindings.push_back(binding);

    binding.binding = 5;                                //binary image
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.push_back(binding);

    std::vector<bool> validd{ true,false,true,true };
    VkUtil::createDescriptorSetLayout(device,bindings,&computeDescriptorSetLayout);
    std::vector<VkDescriptorSetLayout>layouts;
    layouts.push_back(computeDescriptorSetLayout);

    VkUtil::createComputePipeline(device, computeModule, layouts, &computePipelineLayout, &computePipeline);

    //creating the compute pipeline to fill the binary images -----------------------------------------------------
    VkShaderModule binaryModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(binaryComputePath));

    bindings.clear();
    binding.binding = 0;                                //binary compute infos (eg. brush infos)
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(binding);

    binding.binding = 1;                                //density images
    binding.descriptorCount = MAXAMTOF3DTEXTURES;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.push_back(binding);

    binding.binding = 2;                                //binary image
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.push_back(binding);

    validd = { true,false,true };
    VkUtil::createDescriptorSetLayoutPartiallyBound(device, bindings, validd, &binaryComputeDescriptorSetLayout);
    layouts.clear();
    layouts.push_back(binaryComputeDescriptorSetLayout);

    VkUtil::createComputePipeline(device, binaryModule, layouts, &binaryComputePipelineLayout, &binaryComputePipeline);

    //creating the compute pipeline to smooth the binary images -----------------------------------------------------
    computeModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(binarySmoothPath));

    bindings.clear();
    binding.binding = 0;                                //smooth infos
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings.push_back(binding);

    binding.binding = 1;                                //src image
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.push_back(binding);

    binding.binding = 2;                                //dst image
    bindings.push_back(binding);

    VkUtil::createDescriptorSetLayout(device, bindings, &binarySmoothDescriptorSetLayout);
    layouts.clear();
    layouts.push_back(binarySmoothDescriptorSetLayout);

    VkUtil::createComputePipeline(device, computeModule, layouts, &binarySmoothPipelineLayout, &binarySmoothPipeline);

    //creating the compute pipeline to fill the binary iamge with active indices ------------------------------------
    computeModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(activeIndComputePath));

    bindings.clear();
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binding.binding = 0;                                //compute infos
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(binding);

    binding.binding = 1;                                //active indices buffer
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    bindings.push_back(binding);

    binding.binding = 2;                                //indices buffer
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.push_back(binding);

    binding.binding = 3;                                //data buffer
    bindings.push_back(binding);

    binding.binding = 4;                                //dimension arrays
    bindings.push_back(binding);

    binding.binding = 5;                                //binary image
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.push_back(binding);

    VkUtil::createDescriptorSetLayout(device, bindings, &activeIndComputeDescriptorSetLayout);
    layouts.clear();
    layouts.push_back(activeIndComputeDescriptorSetLayout);

    VkUtil::createComputePipeline(device, computeModule, layouts, &activeIndComputePipelineLayout, &activeIndComputePipeline);

    //creating the compute pipeline to copy one entrys into the smoothed binary image ----------------------------------
    computeModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(binaryCopyOnesPath));

    bindings.clear();

    binding.binding = 0;                                //src image
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.push_back(binding);

    binding.binding = 1;                                //dst image
    bindings.push_back(binding);

    VkUtil::createDescriptorSetLayout(device, bindings, &binaryCopyOnesDescriptorSetLayout);
    layouts.clear();
    layouts.push_back(binaryCopyOnesDescriptorSetLayout);

    VkUtil::createComputePipeline(device, computeModule, layouts, &binaryCopyOnesPipelineLayout, &binaryCopyOnesPipeline);
}

void IsoSurfRenderer::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(descriptorSetLayout);
    if (!descriptorSet) {
        VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);
    }
}

void IsoSurfRenderer::updateDescriptorSet()
{
    VkUtil::updateDescriptorSet(device, uniformBuffer, sizeof(UniformBuffer), 0, descriptorSet);
    std::vector<VkImageLayout> layouts(binaryImageView.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    //std::vector<VkImage> binaryImages{ binaryImage };
    //std::vector<VkImageView> binaryImagesView{ binaryImageView };
    std::vector<VkSampler> samplers(binaryImageView.size(), binaryImageSampler);
    VkUtil::updateImageArrayDescriptorSet(device, samplers, binarySmoothView, layouts, 1, descriptorSet);
    VkUtil::updateDescriptorSet(device, brushBuffer, brushByteSize, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);
    layouts = std::vector<VkImageLayout>(3, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    samplers = std::vector<VkSampler>(3, binaryImageSampler);
    VkUtil::updateImageArrayDescriptorSet(device, samplers, dimensionCorrectionViews, layouts, 3, descriptorSet);
}

void IsoSurfRenderer::updateBrushBuffer()
{
    if (drawlistBrushes.empty()) return;

    ////converting the map of brushes to the graphics data structure
    //std::vector<std::vector<std::vector<std::pair<float, float>>>> gpuData;
    //for (auto& brush : brushes) {
    //    for (int axis = 0; axis < brush.second.size(); ++axis) {
    //        if (gpuData.size() <= axis) gpuData.push_back({});
    //        if(brush.second[axis].size()) gpuData[axis].push_back({});
    //        for (auto& minMax : brush.second[axis]) {
    //            gpuData[axis].back().push_back(minMax);
    //        }
    //    }
    //}
    //
    ////get the size for the new buffer
    //uint32_t byteSize = 4 * sizeof(float);        //Standard information + padding
    //byteSize += gpuData.size() * sizeof(float);        //offsets for the axes(offset a1, offset a2, ..., offset an)
    //for (int axis = 0; axis < gpuData.size(); ++axis) {
    //    byteSize += (1 + gpuData[axis].size()) * sizeof(float);        //amtOfBrushes + offsets of the brushes
    //    for (int brush = 0; brush < gpuData[axis].size(); ++brush) {
    //        byteSize += (6 + 2 * gpuData[axis][brush].size()) * sizeof(float);        //brush index(1) + amtOfMinMax(1) + color(4) + space for minMax
    //    }
    //}

    uint32_t byteSize = sizeof(BrushInfos) + drawlistBrushes.size() * 4 * sizeof(float);
    if (brushByteSize >= byteSize) return;        //if the current brush byte size is bigger or equal to the requred byte size simply return. No new allocastion needed

    brushByteSize = byteSize;

    //deallocate too small buffer
    if (brushBuffer) vkDestroyBuffer(device, brushBuffer, nullptr);
    if (brushMemory) vkFreeMemory(device, brushMemory, nullptr);

    VkUtil::createBuffer(device, byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &brushBuffer);
    VkMemoryRequirements memReq = {};
    VkMemoryAllocateInfo allocInfo = {};
    vkGetBufferMemoryRequirements(device, brushBuffer, &memReq);
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &brushMemory);
    vkBindBufferMemory(device, brushBuffer, brushMemory, 0);
}

void IsoSurfRenderer::updateCommandBuffer()
{
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    VkResult err;
    VkUtil::createCommandBuffer(device, commandPool, &commandBuffer);
    VkUtil::transitionImageLayout(commandBuffer, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    std::vector<VkClearValue> clearValues;
    clearValues.push_back(imageBackground);
    VkUtil::beginRenderPass(commandBuffer, clearValues, renderPass, frameBuffer, { imageWidth,imageHeight });

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
    VkDescriptorSet sets[1] = { descriptorSet };
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, sets, 0, nullptr);

    VkViewport viewport = {};                    //description for our viewport for transformation operation after rasterization
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = imageWidth;
    viewport.height = imageHeight;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor = {};                        //description for cutting the rendered result if wanted
    scissor.offset = { 0, 0 };
    scissor.extent = { (uint32_t)imageWidth,(uint32_t)imageHeight };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDrawIndexed(commandBuffer, 3 * 6 * 2, 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    VkUtil::transitionImageLayout(commandBuffer, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    err = vkEndCommandBuffer(commandBuffer);
    check_vk_result(err);

    err = vkDeviceWaitIdle(device);
    check_vk_result(err);
}

bool IsoSurfRenderer::updateDimensionImages(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim)
{
    if (!dimensionCorrectionMemory) {
        VkUtil::create1dImage(device, dimensionCorrectionSize, dimensionCorrectionFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dimensionCorrectionImages);
        VkUtil::create1dImage(device, dimensionCorrectionSize, dimensionCorrectionFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dimensionCorrectionImages + 1);
        VkUtil::create1dImage(device, dimensionCorrectionSize, dimensionCorrectionFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dimensionCorrectionImages + 2);
        VkMemoryRequirements memReq;
        VkMemoryAllocateInfo alloc{};
        vkGetImageMemoryRequirements(device, dimensionCorrectionImages[0], &memReq);
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize += memReq.size;
        vkGetImageMemoryRequirements(device, dimensionCorrectionImages[1], &memReq);
        alloc.allocationSize += memReq.size;
        vkGetImageMemoryRequirements(device, dimensionCorrectionImages[2], &memReq);
        alloc.allocationSize += memReq.size;
        alloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, 0);
        check_vk_result(vkAllocateMemory(device, &alloc, nullptr, &dimensionCorrectionMemory));
        vkBindImageMemory(device, dimensionCorrectionImages[0], dimensionCorrectionMemory, 0);
        vkBindImageMemory(device, dimensionCorrectionImages[1], dimensionCorrectionMemory, alloc.allocationSize / 3);
        vkBindImageMemory(device, dimensionCorrectionImages[2], dimensionCorrectionMemory, alloc.allocationSize / 3 * 2);
        VkUtil::create1dImageView(device, dimensionCorrectionImages[0], dimensionCorrectionFormat, 1, dimensionCorrectionViews.data());
        VkUtil::create1dImageView(device, dimensionCorrectionImages[1], dimensionCorrectionFormat, 1, dimensionCorrectionViews.data() + 1);
        VkUtil::create1dImageView(device, dimensionCorrectionImages[2], dimensionCorrectionFormat, 1, dimensionCorrectionViews.data() + 2);

        VkCommandBuffer command;
        VkUtil::createCommandBuffer(device, commandPool, &command);
        VkUtil::transitionImageLayout(command, dimensionCorrectionImages[0], dimensionCorrectionFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkUtil::transitionImageLayout(command, dimensionCorrectionImages[1], dimensionCorrectionFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkUtil::transitionImageLayout(command, dimensionCorrectionImages[2], dimensionCorrectionFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkUtil::commitCommandBuffer(queue, command);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(device, commandPool, 1, &command);
    }


    if (!PCUtil::vectorEqual(xDim, dimensionCorrectionArrays[0]) || !PCUtil::vectorEqual(yDim, dimensionCorrectionArrays[1]) || !PCUtil::vectorEqual(zDim, dimensionCorrectionArrays[2])) {
        dimensionCorrectionArrays[0] = std::vector<float>(xDim);
        dimensionCorrectionArrays[1] = std::vector<float>(yDim);
        dimensionCorrectionArrays[2] = std::vector<float>(zDim);
        std::vector<float> correction(dimensionCorrectionSize);
        float alpha = 0;
        if (!dimensionCorrectionLinearDim[0]) {
            for (int i = 0; i < dimensionCorrectionSize; ++i) {
                alpha = i / float(dimensionCorrectionSize - 1);
                float axisVal = alpha * xDim.back() + (1 - alpha) * xDim.front();
                correction[i] = PCUtil::getVectorIndex(xDim, axisVal) / (xDim.size() - 1);
            }
            VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, dimensionCorrectionImages[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dimensionCorrectionFormat, dimensionCorrectionSize, 1, 1, correction.data(), correction.size() * sizeof(float));
        }
        if (!dimensionCorrectionLinearDim[1]) {
            for (int i = 0; i < dimensionCorrectionSize; ++i) {
                alpha = i / float(dimensionCorrectionSize - 1);
                float axisVal = alpha * yDim.back() + (1 - alpha) * yDim.front();
                correction[i] = PCUtil::getVectorIndex(yDim, axisVal) / (yDim.size() - 1);
            }
            VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, dimensionCorrectionImages[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dimensionCorrectionFormat, dimensionCorrectionSize, 1, 1, correction.data(), correction.size() * sizeof(float));
        }
        if (!dimensionCorrectionLinearDim[2]) {
            for (int i = 0; i < dimensionCorrectionSize; ++i) {
                alpha = i / float(dimensionCorrectionSize - 1);
                float axisVal = alpha * zDim.back() + (1 - alpha) * zDim.front();
                correction[i] = PCUtil::getVectorIndex(zDim, axisVal) / (zDim.size() - 1);
            }
            VkUtil::uploadImageData(device, physicalDevice, commandPool, queue, dimensionCorrectionImages[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dimensionCorrectionFormat, dimensionCorrectionSize, 1, 1, correction.data(), correction.size() * sizeof(float));
        }
    }

    return true;
}
