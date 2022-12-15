#pragma once
#include <inttypes.h>
#include <iterator>
#include <array>
#include <algorithm>
#include <cassert>
#include <functional>

#define PREFETCH 0
#if PREFETCH
#include <mmintrin.h>
#include <xmmintrin.h>
#define pfval 	64
#define pfval2 	128
#define pfd(x)	_mm_prefetch((const char*)(x + pfval), _MM_HINT_NTA)
#define pf(x)	_mm_prefetch((const char*)(x + i + pfval), _MM_HINT_NTA)
#define pf2d(x)	_mm_prefetch((const char*)(x + pfval2), _MM_HINT_NTA)
#define pf2(x)	_mm_prefetch((const char*)(x + i + pfval2), _MM_HINT_NTA)
#else
#define pfd(x)
#define pf(x)
#define pf2d(x)
#define pf2(x)
#endif

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
		uint64_t mask = ((v >> 63) - 1) | 0x8000000000000000;
		v ^= mask;
		return reinterpret_cast<double&>(v);
	}
};
// helper functions 
template<class T>
inline uint32_t get_bin(const T& val, uint32_t pass, uint32_t radix_size, uint32_t radix_mask){
	return (val >> (pass * radix_size)) & radix_mask;
}
template<uint32_t hist_size, uint32_t passes>
inline std::array<bool, passes> prefix_sum(std::array<std::array<uint32_t, hist_size>, passes>& histograms, size_t size){
	std::array<uint32_t, passes> sum{};
	std::array<bool, passes> sorted{};
	uint32_t t_sum;
	for(int i = 0; i < hist_size; ++i){
		for(int pass = 0; pass < passes; ++pass){
			auto hist_val = histograms[pass][i];
			if(hist_val == size){
				sorted[pass] = true;
				continue;
			}
			t_sum = hist_val + sum[pass];
			histograms[pass][i] = sum[pass] - 1;
			sum[pass] = t_sum;
		}
	}
	return sorted;
}
} // end internal namespace
template<class iter>
struct return_t{
	iter begin;
	iter end;
};

template<int radix_size_pref = -1, class iter, typename transformer_t = internal::transformer<typename std::iterator_traits<iter>::value_type>>
return_t<iter> sort(iter begin, iter end, iter tmp_begin, transformer_t t = {}){
	//static_assert(!std::is_same_v<typename std::iterator_traits<iter>::iterator_category, std::random_access_iterator_tag> && "Iterator type has to be random access to assure continous memory layout");
	typedef decltype(t(*begin)) internal_t;
	constexpr uint32_t radix_size = radix_size_pref > 0 ? radix_size_pref: sizeof(t(*begin)) > 2 ? 11 : 8;
	constexpr uint32_t passes = (sizeof(t(*begin)) * 8 + radix_size - 1) / radix_size;
	constexpr uint32_t hist_size = 1 << radix_size;
	constexpr uint32_t radix_mask = hist_size - 1;
	
	const size_t size = end - begin;
	
	std::array<std::array<uint32_t, hist_size>, passes> histograms{};
	
	// compute the histograms for all passes at once
	auto prev_val = *begin;
	bool all_sorted = true;
	for(iter i = begin; i != end; ++i){
		pfd(&(*i));
		
		auto val = t(*i);
		for(int pass = 0; pass < passes; ++pass){
			uint32_t bin = (val >> (radix_size * pass)) & radix_mask;
			++histograms[pass][bin];
		}
		if(i != begin && all_sorted){
			if(prev_val > *i)
				all_sorted = false;
			prev_val = *i;
		}
	}
	if(all_sorted)
		return {begin, end};
	
	// prefix sum for the histograms
	auto sorted = internal::prefix_sum<hist_size, passes>(histograms, size);

	bool all_one_bin = std::all_of(sorted.begin(), sorted.end(),[](bool b){return b;});
	if(all_one_bin)
		return {begin, end};

	int start_pass = std::find(sorted.begin(), sorted.end(), false) - sorted.begin();
	int last_pass = (sorted.rend() - std::find(sorted.rbegin(), sorted.rend(), false)) - 1;
	assert(start_pass <= last_pass);

	// check for single sort pass -> instantly safe the orignial numbers sorted and return sorted array
	if(start_pass == last_pass){
		auto j = tmp_begin;
		for(iter i = begin; i != end; ++i){
			auto val = *i;
			auto bin = internal::get_bin(t(val), start_pass, radix_size, radix_mask);
			
			pf2d(&(*i));
			j[++histograms[start_pass][bin]] = val;
		}
		return {j, j + size};
	}

	
	// transform input values in the first loop
	auto j = reinterpret_cast<internal_t*>(&(*tmp_begin));
	for(iter i = begin; i != end; ++i){
		auto val = t(*i);
		auto bin = internal::get_bin(val, start_pass, radix_size, radix_mask);
		
		pf2d(&(*i));
		j[++histograms[start_pass][bin]] = val;
	}
	
	// normal radix sorting passes (last pass transforms back to original type)
	internal_t* from = j;
	internal_t* to = reinterpret_cast<internal_t*>(&(*begin));
	for(int pass = start_pass + 1; pass < last_pass; ++pass){
		// check if numbers are sorted (All numbers fall into the same bin for this pass) and continue if so
		if(sorted[pass])
			continue;

		for(size_t i = 0; i < size; ++i){
			auto val = from[i];
			auto bin = internal::get_bin(val, pass, radix_size, radix_mask);
			
			pf2(from);
			to[++histograms[pass][bin]] = val;
		}
		std::swap(from, to);
	}

	// transform sort values back
	iter dst = to == reinterpret_cast<internal_t*>(&(*begin)) ? begin: tmp_begin;
	for(size_t i = 0; i < size; ++i){
		auto val = from[i];
		auto bin = internal::get_bin(val, last_pass, radix_size, radix_mask);

		pf2(from);
		dst[++histograms[last_pass][bin]] = t(val);
	}
	
	return {dst, dst + size};
}

template<int radix_size_pref = -1, typename T>
void sort(std::vector<T>& v){
	std::vector<T> tmp(v.size());
	auto [b, e] = sort<radix_size_pref>(v.begin(), v.end());
	if(&*b != &*v.begin())	// result lies in tmp
		v = std::move(tmp);
}

template<int radix_size_pref = -1, class iter, class user_functor, class transformer_t = internal::transformer<decltype(std::declval<user_functor>()(*std::declval<iter>()))>>
return_t<iter> sort_indirect(iter begin, iter end, iter tmp_begin, user_functor f, transformer_t t = {}){
	constexpr uint32_t radix_size = radix_size_pref > 0 ? radix_size_pref: sizeof(t(f(*begin))) > 2 ? 11 : 8;
	constexpr uint32_t passes = (sizeof(t(f(*begin))) * 8 + radix_size - 1) / radix_size;
	constexpr uint32_t hist_size = 1 << radix_size;
	constexpr uint32_t radix_mask = hist_size - 1;

	const size_t size = end - begin;
	
	std::array<std::array<uint32_t, hist_size>, passes> histograms{};

	// compute the histograms for all passes at once
	auto prev_val = t(f(*begin));
	bool all_sorted = true;
	for(iter i = begin; i != end; ++i){		
		auto val = t(f(*i));
		for(int pass = 0; pass < passes; ++pass){
			uint32_t bin = (val >> (radix_size * pass)) & radix_mask;
			++histograms[pass][bin];
		}
		if(i != begin && all_sorted){
			if(prev_val > val)
				all_sorted = false;
			prev_val = val;
		}
	}
	if(all_sorted)
		return {begin, end};

	// prefix sum for the histograms
	auto sorted = internal::prefix_sum<hist_size, passes>(histograms, size);

	bool all_one_bin = std::all_of(sorted.begin(), sorted.end(),[](bool b){return b;});
	if(all_one_bin)
		return {begin, end};

	int start_pass = std::find(sorted.begin(), sorted.end(), false) - sorted.begin();
	int last_pass = (sorted.rend() - std::find(sorted.rbegin(), sorted.rend(), false)) - 1;
	assert(start_pass <= last_pass);

	iter from = begin;
	iter to = tmp_begin;
	for(int pass = start_pass; pass <= last_pass; ++pass){
		// check if numbers are sorted (All numbers fall into the same bin for this pass) and continue if so
		if(sorted[pass])
			continue;

		for(size_t i = 0; i < size; ++i){
			auto val = from[i];
			auto bin = internal::get_bin(t(f(val)), pass, radix_size, radix_mask);
			
			to[++histograms[pass][bin]] = val;
		}
		std::swap(from, to);
	}

	return {from, from + size};
}

template<int radix_size_pref = -1, typename T, typename user_functor>
void sort_indirect(std::vector<T>& v, user_functor f){
	std::vector<T> tmp(v.size());
	auto [b, e] = sort_indirect<radix_size_pref>(v.begin(), v.end(), tmp.begin(), f);
	if(&*b != &*v.begin())
		v = std::move(tmp);
}
}