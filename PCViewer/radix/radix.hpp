#pragma once
#include <inttypes.h>
#include <iterator>
#include <array>
#include <algorithm>

namespace radix{
namespace internal{
template<typename T>
struct transformer{
	template<typename U>
	U operator()(T v) const{
		static_assert("Transformer for this type not implemented. Provide transformer function for sort");
	}
	template<typename U>
	T operator()(U v) const{
		static_assert("Inverse Transform for this type not implemented. Provide transformer function for sort;");
	}
};

// integer types --------------------------------
template<> struct transformer<uint8_t>{
	uint8_t operator()(uint8_t v) const { return v;}
};
template<> struct transformer<int8_t>{
	uint8_t operator()(int8_t v) const{
		return v ^ 0x80;
	}
	int8_t operator()(uint8_t v) const {
		return v ^ 0x80;
	}
};
template<> struct transformer<uint16_t>{
	uint16_t operator()(uint16_t v) const {return v;}
};
template<> struct transformer<int16_t>{
	uint16_t operator()(int16_t v) const{
		return v ^ 0x8000;
	}
	int16_t operator()(uint16_t v) const{
		return v ^ 0x8000;
	}
};
template<> struct transformer<uint32_t>{
	uint32_t operator()(uint32_t v) const{ return v;}
};
template<> struct transformer<int32_t>{
	uint32_t operator()(int32_t v) const{
		return v ^ 0x80000000;
	}
	int32_t operator()(uint32_t v) const{
		return v ^ 0x80000000;
	}
};
template<> struct transformer<uint64_t>{
	uint64_t operator()(uint64_t v) const{ return v;}
};
template<> struct transformer<int64_t>{
	uint64_t operator()(int64_t v) const{ 
		return v ^ 0x8000000000000000;
	}
	int64_t operator()(uint64_t v) const{
		return v ^ 0x8000000000000000;
	}
};

// floating types --------------------------------
template<> struct transformer<float>{
	uint32_t operator()(float v) const{
		uint32_t& val = reinterpret_cast<uint32_t&>(v);
		uint32_t mask = -int32_t(val >> 31) | 0x80000000;
		return val ^ mask;
	}
	float operator()(uint32_t v) const{
		uint32_t mask = ((v >> 31) - 1) | 0x80000000;
		v ^= mask;
		return reinterpret_cast<float&>(v);
	}
};
template<> struct transformer<double>{
	uint64_t operator()(double v) const{
		uint64_t& val = reinterpret_cast<uint64_t&>(v);
		uint64_t mask = -int64_t(val >> 63) | 0x8000000000000000;
		return val ^ mask;
	}
	double operator()(uint64_t v) const{
		uint32_t mask = ((v >> 63) - 1) | 0x8000000000000000;
		v ^= mask;
		return reinterpret_cast<double&>(v);
	}
};
// helper functions 
template<class T>
inline uint32_t get_bin(const T& val, uint32_t pass, uint32_t radix_size, uint32_t radix_mask){
	return (val >> (pass * radix_size)) & radix_mask;
}
} // end internal namespace
template<class iter>
struct return_t{
	iter begin;
	iter end;
};

template<class iter, typename transformer_t = internal::transformer<typename std::iterator_traits<iter>::value_type>>
return_t<iter> sort(iter begin, iter end, iter tmp_begin, transformer_t t = {}){
	//static_assert(!std::is_same_v<typename std::iterator_traits<iter>::iterator_category, std::random_access_iterator_tag> && "Iterator type has to be random access to assure continous memory layout");
	typedef typename transformer_t::operator(*begin) internal_t;
	constexpr uint32_t radix_size = sizeof(t(*begin)) > 2 ? 11 : 8;
	constexpr uint32_t passes = (sizeof(t(*begin)) * 8 + radix_size - 1) / radix_size;
	constexpr uint32_t hist_size = 1 << radix_size;
	constexpr uint32_t radix_mask = hist_size - 1;
	
	const size_t size = end - begin;
	
	uint32_t hist[hist_size * passes]{};
	uint32_t* hist_start[passes];
	for(int i = 0; i < passes; ++i)	
		hist_start[i] = hist + i * hist_size;
	
	// compute the histograms for all passes at once
	for(iter i = begin; i != end; ++i){
		// TODO implement prefetch
		
		auto val = t(*i);
		for(int pass = 0; pass < passes; ++pass){
			uint32_t bin = (val >> (radix_size * pass)) & radix_mask;
			++hist_start[pass][bin];
		}
	}
	
	// prefix sum for the histograms
	uint32_t sum[passes]{};
	std::array<bool, passes> sorted{};
	bool all_sorted = true;
	uint32_t t_sum;
	for(int i = 0; i < hist_size; ++i){
		for(int pass = 0; pass < passes; ++pass){
			auto hist_val = hist_start[pass][i];
			if(hist_val == size){
				sorted[pass] = true;
				continue;
			}
			t_sum = hist_val + sum[pass];
			hist_start[pass][i] = sum[pass] - 1;
			sum[pass] = t_sum;
			all_sorted = false;
		}
	}

	if(all_sorted)
		return {begin, end};

	int start_pass = std::find(sorted.begin(), sorted.end(), true) - sorted.begin();
	int last_pass = sorted.rend() - std::find(sorted.rbegin(), sorted.rend(), true);

	// check for single sort pass -> instantly safe the orignial numbers sorted and return sorted array
	if(start_pass == last_pass){
		auto j = tmp_begin;
		for(iter i = begin; i != end; ++i, ++j){
			auto val = *i;
			auto bin = internal::get_bin(t(val), start_pass, radix_size, radix_mask);
			// todo prefetch
			j[++hist_start[start_pass][bin]] = val;
		}
		return {j, j + size};
	}

	
	// transform input values in the first loop
	auto j = reinterpret_cast<internal_t*>(&(*tmp_begin));
	for(iter i = begin; i != end; ++i, ++j){
		auto val = t(*i);
		auto bin = internal::get_bin(val, start_pass, radix_size, radix_mask);
		
		// TODO prefetch
		j[++hist_start[start_pass][bin]] = val;
	}
	
	// normal radix sorting passes (last pass transforms back to original type)
	internal_t* from = j;
	internal_t* to = reinterpret_cast<internal_t*>(&(*begin));
	for(int pass = start_pass + 1; pass < last_pass - 1; ++pass){
		// check if numbers are sorted (All numbers fall into the same bin for this pass) and continue if so
		if(sorted[pass])
			continue;

		for(size_t i = 0; i < size; ++i){
			auto val = from[i];
			auto bin = internal::get_bin(val, pass, radix_size, radix_mask);
			// TODO prefetch to preload to
			to[++hist_start[pass][bin]] = val;
		}
		std::swap(from, to);
	}

	// transform sort values back
	for(size_t i = 0; i < size; ++i){
		auto val = from[i];
		auto bin = internal::get_bin(val, last_pass, radix_size, radix_mask);

		//TODO prefetch
		to[++hist_start[last_pass][bin]] = t(val);
	}
	
	if(to == reinterpret_cast<internal_t*>(&(*begin)))
		return {begin, end};
	else
		return {tmp_begin, tmp_begin + size};
}
}