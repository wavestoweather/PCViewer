#pragma once

#include <vector>
#include <memory_view.hpp>

template<typename infos, typename T_array>
struct array_struct: public infos{
	std::vector<T_array> array;
	inline uint32_t size(){
		return sizeof(infos) + sizeof(array[0]) * array.size();
	}
	std::vector<uint8_t> toByteArray(){
		std::vector<uint8_t> bytes(size());
		const uint32_t infoSize = size();
		std::memcpy(bytes.data(), this, infoSize);
		std::memcpy(bytes.data() + infoSize, array.data(), sizeof(array[0] * array.size()));
		return bytes;
	}
    void fillMemory(util::memory_view<uint8_t> memory){
        const uint32_t infoSize = size();
        assert(memory.size() == infoSize);
		std::memcpy(memory.data(), this, infoSize);
		std::memcpy(memory.data() + infoSize, array.data(), sizeof(array[0] * array.size()));
    }
};
