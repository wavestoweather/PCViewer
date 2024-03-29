#include "BubblePlotter.h"

char BubblePlotter::vertPath[] = "shader/nodeVert.spv";
char BubblePlotter::geomPath[] = "shader/nodeGeom.spv";
char BubblePlotter::fragPath[] = "shader/nodeFrag.spv";

BubblePlotter::BubblePlotter(uint32_t width, uint32_t height, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool)
{
    imageWidth = 0;
    imageHeight = 0;
    bubbleIndexSize = 0;
    maxPointSize = 10;
    farClip = 100;
    Fov = 45;
    fovSpeed = 25;
    flySpeed = .5f;
    fastFlyMultiplier = 2.5f;
    rotationSpeed = .25f;
    alphaMultiplier = 1;
    clipping = false;
    normalization = true;
    grey[0] = .5f;
    grey[1] = .5f;
    grey[2] = .5f;
    grey[3] = .2f;
    scale = Scale_Normal;
    scalePointsOnZoom = true;
    layerSpacing = .01f;
    boundingRectMin = glm::vec3(1.6, 0,.7);
    boundingRectMax = glm::vec3(2.0, 6000 ,1.4);
    clippingRectMin = glm::vec3(-INFINITY,-INFINITY,-INFINITY);
    clippingRectMax = glm::vec3(INFINITY,INFINITY,INFINITY);
    attributeActivations = nullptr;
    attributeColors = nullptr;
    attributeTopOffsets = nullptr;
    distribution = std::uniform_int_distribution<int>(0, 35);
    randomEngine = std::default_random_engine();
    posIndices = glm::uvec3(0, 2, 1);
    amtOfAttributes = 0;

    this->physicalDevice = physicalDevice;
    this->device = device;
    this->descriptorPool = descriptorPool;
    this->queue = queue;
    this->commandPool = commandPool;

    renderCommands = VK_NULL_HANDLE;
    imageMemory = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    depthImageMemory = VK_NULL_HANDLE;
    depthImage = VK_NULL_HANDLE;
    imageView = VK_NULL_HANDLE;
    depthImageView = VK_NULL_HANDLE;
    framebuffer = VK_NULL_HANDLE;
    imageSampler = VK_NULL_HANDLE;
    imageDescSet = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    descriptorSetLayout = VK_NULL_HANDLE;
    bufferMemory = VK_NULL_HANDLE;
    sphereVertexBuffer = VK_NULL_HANDLE;
    sphereIndexBuffer = VK_NULL_HANDLE;
    cylinderVertexBuffer = VK_NULL_HANDLE;
    cylinderIndexBuffer = VK_NULL_HANDLE;
    bubbleIndexMemory = VK_NULL_HANDLE;
    bubbleIndexBuffer = VK_NULL_HANDLE;

    ubo = VK_NULL_HANDLE;
    uboMem = VK_NULL_HANDLE;
    uboSet = VK_NULL_HANDLE;
    dataBuffer = VK_NULL_HANDLE;
    dataActivations = VK_NULL_HANDLE;

    cameraPos = glm::vec3(2, 2, 2);
    cameraRot = glm::vec3(0, .78f, 0);
    setupBuffer();
    setupRenderPipeline();
    resizeImage(width, height);
    setupUbo();
    //recordRenderCommands();
    render();
}

BubblePlotter::~BubblePlotter()
{
    if (imageMemory) {
        vkFreeMemory(device, imageMemory, nullptr);
    }
    if (image) {
        vkDestroyImage(device, image, nullptr);
    }
    if (depthImageMemory) {
        vkFreeMemory(device, depthImageMemory, nullptr);
    }
    if (depthImage) {
        vkDestroyImage(device, depthImage, nullptr);
    }
    if (imageView) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    if (depthImageView) {
        vkDestroyImageView(device, depthImageView, nullptr);
    }
    if (framebuffer) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    if (imageSampler) {
        vkDestroySampler(device, imageSampler, nullptr);
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
    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }
    if (bufferMemory) {
        vkFreeMemory(device, bufferMemory, nullptr);
    }
    if (sphereVertexBuffer) {
        vkDestroyBuffer(device, sphereVertexBuffer, nullptr);
    }
    if (sphereIndexBuffer) {
        vkDestroyBuffer(device, sphereIndexBuffer, nullptr);
    }
    if (cylinderVertexBuffer) {
        vkDestroyBuffer(device, cylinderVertexBuffer, nullptr);
    }
    if (cylinderIndexBuffer) {
        vkDestroyBuffer(device, cylinderIndexBuffer, nullptr);
    }
    if (bubbleIndexBuffer) {
        vkDestroyBuffer(device, bubbleIndexBuffer, nullptr);
    }
    if (bubbleIndexMemory) {
        vkFreeMemory(device, bubbleIndexMemory, nullptr);
    }
    if (ubo) {
        vkDestroyBuffer(device, ubo, nullptr);
    }
    if (uboMem) {
        vkFreeMemory(device, uboMem, nullptr);
    }
    if (attributeActivations) {
        delete[] attributeActivations;
    }
    if (attributeColors) {
        delete[] attributeColors;
    }
    if (attributeTopOffsets) {
        delete[] attributeTopOffsets;
    }
}

void BubblePlotter::resizeImage(uint32_t width, uint32_t height)
{
    VkResult err;

    if (!imageSampler) {
        VkUtil::createImageSampler(device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 16, 1, &imageSampler);
    }
    if (width != imageWidth || height != imageHeight) {
        imageWidth = width;
        imageHeight = height;

        //recreating image resources
        if (imageMemory) {
            vkFreeMemory(device, imageMemory, nullptr);
        }
        if (image) {
            vkDestroyImage(device, image, nullptr);
        }
        if (imageView) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        if (depthImageMemory) {
            vkFreeMemory(device, depthImageMemory, nullptr);
        }
        if (depthImage) {
            vkDestroyImage(device, depthImage, nullptr);
        }
        if (depthImageView) {
            vkDestroyImageView(device, depthImageView, nullptr);
        }
        if (framebuffer) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &image);
        VkUtil::createImage(device, imageWidth, imageHeight, VK_FORMAT_D16_UNORM, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &depthImage);

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        VkMemoryAllocateInfo memAlloc = {};
        memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAlloc.allocationSize = memReq.size;
        uint32_t memBits = memReq.memoryTypeBits;
        memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits, 0);

        err = vkAllocateMemory(device, &memAlloc, nullptr, &imageMemory);
        check_vk_result(err);
        
        vkGetImageMemoryRequirements(device, depthImage, &memReq);
        memAlloc.allocationSize = memReq.size;
        memBits = memReq.memoryTypeBits;
        memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits, 0);

        err = vkAllocateMemory(device, &memAlloc, nullptr, &depthImageMemory);
        check_vk_result(err);

        vkBindImageMemory(device, image, imageMemory, 0);
        vkBindImageMemory(device, depthImage, depthImageMemory, 0);

        VkUtil::createImageView(device, image, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &imageView);
        VkUtil::createImageView(device, depthImage, VK_FORMAT_D16_UNORM, 1, VK_IMAGE_ASPECT_DEPTH_BIT, &depthImageView);
        std::vector<VkImageView> views;
        views.push_back(imageView);
        //views.push_back(depthImageView);
        VkUtil::createFrameBuffer(device, renderPass, views, imageWidth, imageHeight, &framebuffer);

        //transforming the imagelayout to be readable
        VkCommandBuffer command;
        VkUtil::createCommandBuffer(device, commandPool, &command);
        VkUtil::transitionImageLayout(command, image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkUtil::transitionImageLayout(command, depthImage, VK_FORMAT_D16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        vkEndCommandBuffer(command);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &command;

        err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        check_vk_result(err);
    }
}

void BubblePlotter::addSphere(float r, glm::vec4 color, glm::vec3 pos)
{
    gSphere cur;
    cur.radius = r;
    cur.color = color;
    cur.pos = pos;
    //TODO: create descset

    spheres.push_back(cur);
    recordRenderCommands();
}

void BubblePlotter::addCylinder(float r, float length, glm::vec4 color, glm::vec3 pos)
{
    gCylinder cur;
    cur.radius = r;
    cur.length = length;
    cur.color = color;
    cur.pos = pos;
    //TODO: create descriptor set

    cylinders.push_back(cur);
    recordRenderCommands();
}

void BubblePlotter::setBubbleData(std::vector<uint32_t>& indices, std::vector<std::string>& attributeNames, std::vector<std::pair<float, float>> attributesMinMax, const Data& data, VkBuffer gData, VkBufferView activeData, uint32_t amtOfAttributes, uint32_t amtOfData)
{
    if(amtOfAttributes != this->amtOfAttributes){
        if (attributeActivations) {
            delete[] attributeActivations;
        }
        if (attributeColors) {
            delete[] attributeColors;
        }
        if (attributeTopOffsets) {
            delete[] attributeTopOffsets;
        }
        if (attributeMinMaxValues.size()) {
            attributeMinMaxValues.clear();
        }
        this->amtOfAttributes = amtOfAttributes;
        this->attributeActivations = new bool[amtOfAttributes];
        this->attributeColors = new glm::vec4[amtOfAttributes];
        this->attributeTopOffsets = new float[amtOfAttributes];
        int offsetCounter = 0;
        float offsetStep = 1.0f / (amtOfAttributes - 1);
        for (int i = 0; i < amtOfAttributes; ++i) {
            this->attributeActivations[i] = i != posIndices.x && i != posIndices.y && i != posIndices.z;
            this->attributeMinMaxValues.push_back(attributesMinMax[i]);
            float hue = distribution(randomEngine) * 10;
            hsl randCol = { hue,.5f,.6f };
            rgb col = hsl2rgb(randCol);
            this->attributeColors[i] = glm::vec4(col.r, col.g, col.b, .8f);
            this->attributeTopOffsets[i] = offsetCounter * offsetStep;
            ++offsetCounter;
        }
    }
    this->attributeNames = attributeNames;
    this->dataActivations = activeData;
    this->attributeScales = std::vector<BubblePlotter::Scale>(amtOfAttributes, BubblePlotter::Scale_Normal);        //on setting bubble data the scale is reset
    this->dataBuffer = gData;
    amtOfDatapoints = amtOfData;
    //setting the bounding boxes min and max value automatically
    boundingRectMin.x = attributesMinMax[posIndices.x].first;
    boundingRectMin.y = attributesMinMax[posIndices.y].first;
    boundingRectMin.z = attributesMinMax[posIndices.z].first;
    boundingRectMax.x = attributesMinMax[posIndices.x].second;
    boundingRectMax.y = attributesMinMax[posIndices.y].second / 2;        //double scaling in height, as the dots have to be able to be printed in between layers
    boundingRectMax.z = attributesMinMax[posIndices.z].second;
    
    VkResult err;
    if (bubbleIndexMemory) {
        vkFreeMemory(device, bubbleIndexMemory, nullptr);
    }
    if (bubbleIndexBuffer) {
        vkDestroyBuffer(device, bubbleIndexBuffer, nullptr);
    }

    bubbleIndexSize = indices.size()*2;
    VkUtil::createBuffer(device, indices.size() * 2 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &bubbleIndexBuffer);
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, bubbleIndexBuffer, &memReq);

    VkMemoryAllocateInfo memAlloc = {};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    uint32_t memBits = memReq.memoryTypeBits;
    memAlloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    err = vkAllocateMemory(device, &memAlloc, nullptr, &bubbleIndexMemory);
    check_vk_result(err);

    vkBindBufferMemory(device, bubbleIndexBuffer, bubbleIndexMemory, 0);

    this->data = &data;
    this->indices = &indices;
    updateRenderOrder();

    setupUbo();
    recordRenderCommands();
    render();
}

void BubblePlotter::updateRenderOrder()
{
    std::vector<uint32_t> indCopy(*indices);
    //sorting the bubble instances for positive render order
    auto pI = posIndices;
    const Data* dat = data;
    std::sort(indCopy.begin(), indCopy.end(), [dat, pI](uint32_t a, uint32_t b) {float lonA = (*dat)(a,pI.x);
    float lonB = (*dat)(b,pI.x);
    float latA = (*dat)(a,pI.y);
    float latB = (*dat)(b,pI.y);
    float altA = (*dat)(a,pI.z);
    float altB = (*dat)(b,pI.z);
    if (altA > altB) return true;
    else if (altA < altB) return false;
    else {
        if (latA > latB)return true;
        else if (latA < latB)return false;
        else return lonA > lonB;
    }});
    indCopy.insert(indCopy.end(), indCopy.rbegin(), indCopy.rend());

    VkUtil::uploadData(device, bubbleIndexMemory, 0, indCopy.size() * sizeof(uint32_t), indCopy.data());
}

void BubblePlotter::render()
{
    if (!dataBuffer) return;

    VkResult err;

    uint32_t uboByteSize = sizeof(Ubo) + amtOfAttributes * 8 * sizeof(float);
    unsigned char* uboBytes = new unsigned char[uboByteSize];
    Ubo& ubo = *((Ubo*)uboBytes);
    ubo.cameraPos = glm::vec4(cameraPos, maxPointSize);
    glm::mat4 proj = glm::perspective(glm::radians(Fov), (float)imageWidth / (float)imageHeight, 0.001f, farClip);
    proj[1][1] *= -1;
    glm::mat4 view = glm::transpose(glm::eulerAngleY(cameraRot.y)*glm::eulerAngleX(cameraRot.x)) * glm::translate(glm::mat4(1.0),-cameraPos);
    ubo.mvp = proj * view;
    ubo.alphaMultiplier = alphaMultiplier;
    ubo.amtOfAttributes = amtOfAttributes;
    ubo.FoV = Fov;
    ubo.clipNormalize = (clipping << 1) | normalization;
    ubo.grey = glm::vec4(grey[0], grey[1], grey[2], grey[3]);
    ubo.posIndices = glm::vec4(posIndices, 0);
    ubo.relative = scalePointsOnZoom;
    ubo.offset = layerSpacing;
    ubo.boundingRectMin = boundingRectMin;
    ubo.boundingRectMax = boundingRectMax;
    ubo.clippingRectMin = clippingRectMin;
    ubo.clippingRectMax = clippingRectMax;
    float* attributeInfos = (float*)(uboBytes + sizeof(Ubo));
    for (int i = 0; i < amtOfAttributes; ++i) {
        attributeInfos[i * 8] = attributeColors[i].x;
        attributeInfos[i * 8 + 1] = attributeColors[i].y;
        attributeInfos[i * 8 + 2] = attributeColors[i].z;
        attributeInfos[i * 8 + 3] = attributeColors[i].w;
        attributeInfos[i * 8 + 4] = (attributeActivations[i]) ? attributeTopOffsets[i] : -1;
        attributeInfos[i * 8 + 5] = attributeMinMaxValues[i].first;
        attributeInfos[i * 8 + 6] = attributeMinMaxValues[i].second;
        attributeInfos[i * 8 + 7] = attributeScales[i];
    }
    void* d;
    vkMapMemory(device, uboMem, 0, uboByteSize, 0, &d);
    memcpy(d, uboBytes, uboByteSize);
    vkUnmapMemory(device, uboMem);

    //getting the right data ordering for the viewing direction
    glm::vec4 front = glm::eulerAngleY(cameraRot.y) * glm::eulerAngleX(cameraRot.x) * glm::vec4(0, 0, -1, 0);
    float posMax = std::fmax(std::fmax(front.x,front.y),front.z);
    front *= -1;
    float negMax = std::fmax(std::fmax(front.x, front.y), front.z);

    //submitting the command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.commandBufferCount = 1;
    if (posMax > negMax) submitInfo.pCommandBuffers = &renderCommands;
    else submitInfo.pCommandBuffers = &inverseRenderCommands;

    err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    check_vk_result(err);
    err = vkQueueWaitIdle(queue);
    check_vk_result(err);

    delete[] uboBytes;
}

void BubblePlotter::updateCameraPos(CamNav::NavigationInput input, float deltaT)
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

    if (input.mouseScrollDelta) {    //mouse wheel input
        Fov -= fovSpeed * input.mouseScrollDelta * deltaT;
    }
}

void BubblePlotter::setPointScale(Scale scale)
{
    maxPointSize = scale;
}

VkSampler BubblePlotter::getImageSampler()
{
    return imageSampler;
}

VkImageView BubblePlotter::getImageView()
{
    return imageView;
}

void BubblePlotter::setImageDescSet(VkDescriptorSet desc)
{
    imageDescSet = desc;
}

VkDescriptorSet BubblePlotter::getImageDescSet()
{
    return imageDescSet;
}

void BubblePlotter::setupRenderPipeline()
{
    VkShaderModule shaderModules[5] = {};
    //the vertex shader for the pipeline
    std::vector<char> vertexBytes = PCUtil::readByteFile(vertPath);
    shaderModules[0] = VkUtil::createShaderModule(device, vertexBytes);
    //the geometry shader
    std::vector<char> geometryBytes = PCUtil::readByteFile(geomPath);
    shaderModules[3] = VkUtil::createShaderModule(device, geometryBytes);
    //the fragment shader for the pipeline
    std::vector<char> fragmentBytes = PCUtil::readByteFile(fragPath);
    shaderModules[4] = VkUtil::createShaderModule(device, fragmentBytes);

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
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 1;
    bindings.push_back(uboLayoutBinding);

    uboLayoutBinding.binding = 2;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    bindings.push_back(uboLayoutBinding);

    VkUtil::createDescriptorSetLayout(device, bindings, &descriptorSetLayout);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    descriptorSetLayouts.push_back(descriptorSetLayout);

    VkUtil::createRenderPass(device, VkUtil::PASS_TYPE_COLOR_OFFLINE, &renderPass);

    VkUtil::createPipeline(device, &vertexInputInfo, imageWidth, imageHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, &depthStencil, &blendInfo, descriptorSetLayouts, &renderPass, &pipelineLayout, &pipeline);
}

void BubblePlotter::setupBuffer()
{
    VkResult err;

    Sphere s(1, 20);
    Cylinder c(1, 2, 20);

    amtOfIdxSphere = s.getIndexBuffer(0).size();
    amtOfIdxCylinder = c.getIndexBuffer(0).size();

    //creating the buffer
    VkUtil::createBuffer(device, s.getVertexBuffer().size() * sizeof(Shape::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &sphereVertexBuffer);
    VkUtil::createBuffer(device, s.getIndexBuffer(0).size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &sphereIndexBuffer);
    VkUtil::createBuffer(device, c.getVertexBuffer().size() * sizeof(Shape::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &cylinderVertexBuffer);
    VkUtil::createBuffer(device, c.getIndexBuffer(0).size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &cylinderIndexBuffer);

    //getting memory requirements and allocating memory
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, sphereVertexBuffer, &memReq);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    uint32_t memBits = memReq.memoryTypeBits;
    uint32_t sphereIndBufOffset = memReq.size;

    vkGetBufferMemoryRequirements(device, sphereIndexBuffer, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    uint32_t cylinderVertBufOffset = allocInfo.allocationSize;

    vkGetBufferMemoryRequirements(device, cylinderVertexBuffer, &memReq);
    allocInfo.allocationSize += memReq.size;
    memBits |= memReq.memoryTypeBits;
    uint32_t cylinderIndBufOffset = allocInfo.allocationSize;

    vkGetBufferMemoryRequirements(device, cylinderIndexBuffer, &memReq);
    allocInfo.allocationSize += memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memBits | memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    err = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    check_vk_result(err);

    //binding buffer to memory
    vkBindBufferMemory(device, sphereVertexBuffer, bufferMemory, 0);
    vkBindBufferMemory(device, sphereIndexBuffer, bufferMemory, sphereIndBufOffset);
    vkBindBufferMemory(device, cylinderVertexBuffer, bufferMemory, cylinderVertBufOffset);
    vkBindBufferMemory(device, cylinderIndexBuffer, bufferMemory, cylinderIndBufOffset);

    //uploading the vertex and indexbuffers to the graphicscard
    char* data = new char[allocInfo.allocationSize];
    memcpy(data, s.getVertexBuffer().data(), s.getVertexBuffer().size() * sizeof(Shape::Vertex));
    memcpy(data + sphereIndBufOffset, s.getIndexBuffer(0).data(), s.getIndexBuffer(0).size() * sizeof(uint32_t));
    memcpy(data + cylinderVertBufOffset, c.getVertexBuffer().data(), c.getVertexBuffer().size() * sizeof(Shape::Vertex));
    memcpy(data + cylinderIndBufOffset, c.getIndexBuffer(0).data(), c.getIndexBuffer(0).size() * sizeof(uint32_t));

    void* d;
    vkMapMemory(device, bufferMemory, 0, allocInfo.allocationSize, 0, &d);
    memcpy(d, data, allocInfo.allocationSize);
    vkUnmapMemory(device, bufferMemory);

    delete[] data;
}

void BubblePlotter::recordRenderCommands()
{
    if (!dataBuffer) return;

    if (renderCommands) {
        VkCommandBuffer buffers[2] = { renderCommands,inverseRenderCommands };
        vkFreeCommandBuffers(device, commandPool, 2, buffers);
    }

    VkResult err;
    VkUtil::createCommandBuffer(device, commandPool, &renderCommands);

    VkUtil::transitionImageLayout(renderCommands, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    std::vector<VkClearValue> clearValues;
    clearValues.push_back({ 0,0,0,1 });
    clearValues.push_back({ 1.0f });
    VkUtil::beginRenderPass(renderCommands, clearValues, renderPass, framebuffer, { imageWidth,imageHeight });
    VkViewport viewport = {};                    //description for our viewport for transformation operation after rasterization
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = imageWidth;
    viewport.height = imageHeight;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(renderCommands, 0, 1, &viewport);

    VkRect2D scissor = {};                        //description for cutting the rendered result if wanted
    scissor.offset = { 0, 0 };
    scissor.extent = { (uint32_t)imageWidth,(uint32_t)imageHeight };
    vkCmdSetScissor(renderCommands, 0, 1, &scissor);

    vkCmdBindPipeline(renderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindIndexBuffer(renderCommands, bubbleIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(renderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uboSet, 0, nullptr);
    vkCmdDraw(renderCommands, bubbleIndexSize >> 1, 1, 0, 0);

    vkCmdEndRenderPass(renderCommands);

    VkUtil::transitionImageLayout(renderCommands, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    err = vkEndCommandBuffer(renderCommands);
    check_vk_result(err);

    //recording the inverse command buffer
    VkUtil::createCommandBuffer(device, commandPool, &inverseRenderCommands);

    VkUtil::transitionImageLayout(inverseRenderCommands, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkUtil::beginRenderPass(inverseRenderCommands, clearValues, renderPass, framebuffer, { imageWidth,imageHeight });
    vkCmdSetViewport(inverseRenderCommands, 0, 1, &viewport);
    vkCmdSetScissor(inverseRenderCommands, 0, 1, &scissor);

    vkCmdBindPipeline(inverseRenderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindIndexBuffer(inverseRenderCommands, bubbleIndexBuffer, (bubbleIndexSize >> 1) * sizeof(uint32_t), VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(inverseRenderCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &uboSet, 0, nullptr);
    vkCmdDraw(inverseRenderCommands, bubbleIndexSize >> 1, 1, 0, 0);

    vkCmdEndRenderPass(inverseRenderCommands);

    VkUtil::transitionImageLayout(inverseRenderCommands, image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    err = vkEndCommandBuffer(inverseRenderCommands);
    check_vk_result(err);
}

void BubblePlotter::setupUbo()
{
    if (ubo != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, ubo, nullptr);
        vkFreeMemory(device, uboMem, nullptr);
        vkFreeDescriptorSets(device, descriptorPool, 1, &uboSet);
    }

    VkResult err;

    uint32_t uboByteSize = sizeof(Ubo) + amtOfAttributes * 8 * sizeof(float);
    VkUtil::createBuffer(device, uboByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ubo);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, ubo, &memReq);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    err = vkAllocateMemory(device, &allocInfo, nullptr, &uboMem);
    check_vk_result(err);

    vkBindBufferMemory(device, ubo, uboMem, 0);

    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(descriptorSetLayout);
    VkUtil::createDescriptorSets(device, layouts, descriptorPool, &uboSet);

    VkUtil::updateDescriptorSet(device, ubo, uboByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uboSet);
    if (dataBuffer)VkUtil::updateDescriptorSet(device, dataBuffer, VK_WHOLE_SIZE, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uboSet);
    if (dataActivations)VkUtil::updateTexelBufferDescriptorSet(device, dataActivations, 2, uboSet);
}