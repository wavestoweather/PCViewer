#pragma once

#include <vector>
#include <memory_view.hpp>
#include <cstring>

namespace structures{
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

template<typename T_info, typename T_array>
struct dynamic_struct{
	dynamic_struct(size_t size = 0): _storage(sizeof(T_info) + size * sizeof(T_array)), _info_ptr(reinterpret_cast<T_info*>(_storage.data())), _array_ptr(reinterpret_cast<T_array*>( _storage.data() + sizeof(T_info))), _array_size(size){}

	T_info& info() {return *_info_ptr;}

	util::memory_view<const uint8_t> data() {return util::memory_view<uint8_t>(_storage);}

	//void push_back(const T_array& e) {_storage.insert(_storage.end(), sizeof(T_array), {}); ++array_size; _info_ptr = reinterpret_cast<T_info*>(_storage.data()); _array_ptr = reinterpret_cast<T_array*>(_storage.data() + sizeof(T_info)); _array_ptr[_array_size - 1] = e};
	//void push_back(T_array&& e) {_storage.insert(_storage.end(), sizeof(T_array), {}); ++array_size; _info_ptr = reinterpret_cast<T_info*>(_storage.data()); _array_ptr = reinterpret_cast<T_array*>(_storage.data() + sizeof(T_info)); _array_ptr[_array_size - 1] = std::move(e)};

	T_info* operator->() {return _info_ptr;}
	T_array& operator[](size_t i) {return _array_ptr[i];}
	const T_array& operator[](size_t i) const {return _array_ptr[i];}
private:
	std::vector<uint8_t> 	_storage;
	T_info* 				_info_ptr;
	T_array*				_array_ptr;
	size_t					_array_size;
};
}