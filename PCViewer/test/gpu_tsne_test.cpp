#include "test_commons.hpp"
#include <tsne_compute.hpp>
#include <vk_context.hpp>
#include <vk_initializers.hpp>

struct test_result{
    static const int success = 0;
    static const int flag_wrong = 1;
};

void vulkan_default_init(){
    // vulkan init
    std::vector<const char*> instance_extensions;
    instance_extensions.push_back("VK_KHR_get_physical_device_properties2");
    std::vector<const char*> instance_layers;
    instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push_back("VK_LAYER_KHRONOS_validation");

    std::vector<const char*> device_extensions{ VK_KHR_MAINTENANCE3_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, /*VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME*/ VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};
    VkPhysicalDeviceVulkan12Features vk_12_features = util::vk::initializers::physicalDeviceVulkan12Features();
    vk_12_features.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDevice16BitStorageFeatures vk_16bit_features = util::vk::initializers::physicalDevice16BitStorageFeatures();
    vk_16bit_features.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT float_feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    float_feat.shaderBufferFloat32Atomics = VK_TRUE;
    VkPhysicalDeviceFeatures2 device_features = util::vk::initializers::physicalDeviceFeatures2();
    device_features.pNext = &vk_12_features;
    vk_12_features.pNext = &vk_16bit_features;
    vk_16bit_features.pNext = &float_feat;
    int physical_device_index = -1;
    structures::VkContextInitInfo vk_init{
        physical_device_index,
        VK_API_VERSION_1_2,
        "PCViewer",
        instance_layers,
        instance_extensions,
        device_extensions,
        device_features
    };
    auto chosen_gpu = globals::vk_context.init(vk_init);
}

int test_pipeline_creation(){
    auto& pipeline = pipelines::tsne_compute::instance();
    //pipeline.record({}, {});
    return test_result::success;
}

int gpu_tsne_test(int argc, char** const argv){
    vulkan_default_init();
    check_res(test_pipeline_creation());

    std::cout << "[info] gpu_tsne_test successful" << std::endl;
    globals::vk_context.cleanup();
    return test_result::success;
}