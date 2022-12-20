#pragma once
#include <memory_view.hpp>

#define DEFAULT_EQUALS(struct) bool operator==(const struct& o) const {return util::memory_view<const uint32_t>(util::memory_view(*this)).equal_data(util::memory_view(o));}