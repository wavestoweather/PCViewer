#pragma once
#include <robin_hood.h>
#include <string>
#include <image_info.hpp>

template<> struct std::hash<VkSamplerCreateInfo>{
    size_t operator()(const VkSamplerCreateInfo & x) const
    {
        return util::memory_view<const uint32_t>(util::memory_view<const VkSamplerCreateInfo>(x)).data_hash();
    }
};
template<> struct std::equal_to<VkSamplerCreateInfo>{
    bool operator()(const VkSamplerCreateInfo & l, const VkSamplerCreateInfo & r) const
    {
        return util::memory_view<const uint32_t>(util::memory_view<const VkSamplerCreateInfo>(l)).equal_data(util::memory_view<const uint32_t>(util::memory_view<const VkSamplerCreateInfo>(r)));
    }
};

namespace structures{
struct persistent_samplers{
    VkSampler get(const VkSamplerCreateInfo& sampler_info);
private:
    robin_hood::unordered_map<VkSamplerCreateInfo, VkSampler> _samplers;
};
}

namespace globals{
extern structures::persistent_samplers persistent_samplers;
}