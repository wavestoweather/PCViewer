#pragma once
#include <imgui.h>
#include <filesystem>
#include <memory_view.hpp>
#include <vk_initializers.hpp>
#include <vk_util.hpp>
#include "../imgui_file_dialog/CustomFont.cpp"

namespace util{
namespace imgui{
inline void load_fonts(std::string_view font_folder, util::memory_view<float> font_sizes){
    if(!std::filesystem::exists(font_folder)){
        std::cout << "[warning] Font folder " << font_folder << " could not be found. Only standard font will be available" << std::endl;
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig font_conf{};
	font_conf.OversampleH = 2;
	font_conf.OversampleV = 2;
    ImWchar icons_ranges[] = { ICON_MIN_IGFD, ICON_MAX_IGFD, 0 };
	ImFontConfig icons_config{}; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    for(const auto& entry: std::filesystem::directory_iterator(font_folder)){
        if(entry.is_regular_file() && entry.path().has_extension() && entry.path().extension().string() == ".ttf"){
            // found regular file
            for(float size: font_sizes){
                io.Fonts->AddFontFromFileTTF(entry.path().c_str(), size, &font_conf, io.Fonts->GetGlyphRangesDefault());
                io.Fonts->AddFontFromMemoryCompressedBase85TTF(FONT_ICON_BUFFER_NAME_IGFD, size, &icons_config, icons_ranges);
            }
        }
    }
}

inline VkDescriptorPool create_desriptor_pool(){
    std::vector<VkDescriptorPoolSize> pool_sizes{
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    auto pool_info = util::vk::initializers::descriptorPoolCreateInfo(pool_sizes, 1000 * pool_sizes.size(), VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
    return util::vk::create_descriptor_pool(pool_info);
}

}
}