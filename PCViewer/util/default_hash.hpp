#pragma once
#include <memory_view.hpp>

// uses memory view to calculate the default hash, needs the struct to be 4 aligned
#define DEFAULT_HASH(typename) static_assert(sizeof(typename) % sizeof(uint32_t) == 0); template<> struct std::hash<typename>{ size_t operator()(const typename& x) const{ return util::memory_view<const uint32_t>(util::memory_view(x)).data_hash(); }}
