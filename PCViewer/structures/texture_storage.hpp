#pragma once
#include <robin_hood.h>
#include <string>
#include <image_info.hpp>

namespace structures{
struct texture{
    image_info  image;
    VkImageView image_view;

    bool operator==(const texture& o) const {return image == o.image && image_view == o.image_view;}
};
}

namespace globals{
extern robin_hood::unordered_map<std::string, structures::texture> textures;
}