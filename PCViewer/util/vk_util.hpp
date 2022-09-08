#pragma once
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_context.hpp>
#include <vk_initializers.hpp>
#include <stdexcept>
#include <ranges.hpp>

namespace util{
inline void check_vk_result(VkResult err){
    if(err > 0) std::cout << "VkResult " << string_VkResult(err);
    if(err < 0) throw std::runtime_error(std::string("VkResult ") + string_VkResult(err));
}

namespace vk{
template<class T>
struct feature_wrapper{
    T feature;
    std::vector<uint8_t> next_storage;
};
namespace internal{
    template<class T>
    inline void fill_next(void**& curNext, std::vector<uint8_t>*& curStorage){
        void* next = *curNext;
        auto cur = std::vector<uint8_t>(sizeof(feature_wrapper<T>));
        std::memcpy(cur.data(), next, sizeof(T));
        *curNext = cur.data();
        *curStorage = std::move(cur);
        curNext = &(reinterpret_cast<T*>(curStorage->data())->pNext);
        curStorage = &(reinterpret_cast<feature_wrapper<T>*>(curStorage->data())->next_storage);
    }
}

inline feature_wrapper<VkPhysicalDeviceFeatures2> copy_features(const VkPhysicalDeviceFeatures2& in){
    feature_wrapper<VkPhysicalDeviceFeatures2> start{};
    start.feature = in;
    void** curNext = &start.feature.pNext;
    std::vector<uint8_t>* curStorage = &start.next_storage;
    while(*curNext != nullptr){
        void* next = *curNext;
        VkStructureType nextType = *reinterpret_cast<VkStructureType*>(next);
        switch(nextType){
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
                internal::fill_next<VkPhysicalDeviceVulkan11Features>(curNext, curStorage);
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                internal::fill_next<VkPhysicalDeviceVulkan12Features>(curNext, curStorage);
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
                internal::fill_next<VkPhysicalDeviceVulkan12Features>(curNext, curStorage);
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT:
                internal::fill_next<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(curNext, curStorage);
                break;
            default:
            throw std::runtime_error(std::string("util::copy_features() Unhandled feature type in pNext chain: ") + string_VkStructureType(nextType));
        }
    }
    return std::move(start);
}

template<class T>
inline bool all_features_available(const T& available, const T& required){
    const VkBool32* bool_start_avail = reinterpret_cast<const VkBool32*>(&available.pNext) + sizeof(available.pNext) / sizeof(VkBool32);
    const VkBool32* bool_end_avail = reinterpret_cast<const VkBool32*>(&available) + sizeof(T) / sizeof(VkBool32);
    const VkBool32* bool_start_req = reinterpret_cast<const VkBool32*>(&required.pNext) + sizeof(required.pNext) / sizeof(VkBool32);

    int bool_count = bool_end_avail - bool_start_avail;
    assert(bool_count > 0 && bool_count < 150);
    for(int i: i_range(bool_count)){
        if(bool_start_req[i] && !bool_start_avail[i])
            return false;
    }
    return true;
}

inline structures::buffer_info create_buffer(const VkBufferCreateInfo& buffer_info, const VmaAllocationCreateInfo alloc_info,/*out: info for allocated mem*/ VmaAllocationInfo* out_alloc_info = {}){
    structures::buffer_info info{};
    auto res = vmaCreateBuffer(globals::vk_context.allocator, &buffer_info, &alloc_info, &info.buffer, &info.allocation, out_alloc_info);
    check_vk_result(res);
    return info;
}

inline structures::image_info create_image(const VkImageCreateInfo& image_info, const VmaAllocationCreateInfo& alloc_info, /*out: info for allocated mem*/ VmaAllocationInfo* out_alloc_info = {}){
    structures::image_info info{};
    auto res = vmaCreateImage(globals::vk_context.allocator, &image_info, &alloc_info, &info.image, &info.allocation, out_alloc_info);
    check_vk_result(res);
    return info;
}

inline VkImageView create_image_view(const VkImageViewCreateInfo& view_info){
    VkImageView view;
    auto res = vkCreateImageView(globals::vk_context.device, &view_info, globals::vk_context.allocation_callbacks, &view);
    check_vk_result(res);
    return view;
}

inline structures::image_info create_image_with_view(const VkImageCreateInfo& image_info, const VmaAllocationCreateInfo& alloc_info, /*out: info for allocated mem*/ VmaAllocationInfo* out_alloc_info = {}){
    auto info = create_image(image_info, alloc_info, out_alloc_info);
    auto view_info = initializers::imageViewCreateInfo(info.image, static_cast<VkImageViewType>(image_info.imageType), image_info.format);
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.subresourceRange.levelCount = 1;
    info.image_view = create_image_view(view_info);
    return info;
}
}
}