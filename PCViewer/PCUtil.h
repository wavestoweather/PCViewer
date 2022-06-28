#ifndef PCUtil_H
#define PCUtil_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <condition_variable>
#include <sstream>
#include <algorithm>
#include "imgui/imgui.h"
#include "Data.hpp"
#include "Attribute.hpp"

template<typename T>
std::stringstream& operator>>(std::stringstream& out, std::vector<T>& v);
template<typename T>
std::stringstream& operator<<(std::stringstream& out, std::vector<T>& v);

class PCUtil {
public:
	static std::vector<char> readByteFile(const std::string_view& filename);
	static void hexdump(const void* ptr, int buflen);
	static void numdump(const float* ptr, int len, bool lineNumber = false);
	static void numdump(const int* ptr, int len);
	static void matrixdump(const std::vector<std::vector<int>>& matrix);
	static void matrixdump(const std::vector<std::vector<double>>& matrix);
	static float getVectorIndex(const std::vector<float>& values, float v);
	static bool vectorEqual(const std::vector<float>& a, const std::vector<float>& b);
	static float distance2(const ImVec2& a, const ImVec2& b);
	static float distance(const ImVec2& a, const ImVec2& b);
	static bool compareStringFormat(const std::string_view& s, const std::string_view& form);
	static std::vector<QueryAttribute> queryNetCDF(const std::string_view& filename);
	static std::vector<int> checkAttributes(std::vector<std::string>& a, std::vector<Attribute>& ref);
	static Data openNetCdf(const std::string_view& filename, /*inout*/ std::vector<Attribute>& attributes, const std::vector<QueryAttribute>& queryAttributes);
	template<typename T>
	static std::string toReadableString(const std::vector<T>& v){
		std::stringstream out;
		out << "[";
		for(const auto& e: v){
			out << e << ",";
		}
		std::string o = out.str();
		o.back() = ']';
		return o;
	};
	template<typename T>
	static std::vector<T> fromReadableString(const std::string& s){
		if(s[0] != '[' || s.back() != ']')
			throw std::runtime_error{"PCUtil::fromReadableString(): Parsing vector from string failed because the string is not sourrounded by brackets: " + s};
		std::vector<T> res;
		std::string_view elements = std::string_view{s}.substr(1, s.size() - 2);
		auto curStart = elements.begin();
		while(curStart < elements.end()){
			auto nextStart = curStart + 1;
			while(*nextStart != ',' && nextStart < elements.end()){
				if(*nextStart == '['){ // inner vector
					int c = 0;
					while(c > 0){
						if(*nextStart == '[')
							c++;
						if(*nextStart == ']')
							c--;
						++nextStart;
					}
				}
				if(*nextStart != ',')
					++nextStart;
			}
			T e;
			std::stringstream cur(std::string(curStart, nextStart));
			cur >> e;
			res.push_back(e);
			curStart = nextStart + 1;
		}
		return res;
	};

	class Stopwatch{
		public:
		Stopwatch(std::ostream& stream, const std::string_view& displayName);
		~Stopwatch();
		private:
		std::ostream& _ostream;
		std::string _name;
		std::chrono::high_resolution_clock::time_point _start;
	};

	class AverageWatch{
		public:
		AverageWatch(float& average, uint32_t& count);
		~AverageWatch();
		AverageWatch& operator=(const PCUtil::AverageWatch & o);
		private:
		float& _a;
		uint32_t& _c;
		std::chrono::high_resolution_clock::time_point _start;
	};

	class Semaphore{
		std::mutex _mutex;
		std::condition_variable _cv;
		unsigned long _count = 0;
	public:
		void release();
		void releaseN(int n);
		void acquire();
	};

	// custom priority queue implementation to be able to access the underlying container
	template<class T, class Container = std::vector<T>, class Compare = std::less<typename Container::value_type>>
	class PriorityQueue{
	public:
		const Container& container() const {return _container;};
		bool empty() const {return _container.empty();};
		size_t size() const {return _container.size();};
		const T& top() const{return _container[0];};
		void push(const T& value){
			_container.push_back(value);
			std::push_heap(_container.begin(), _container.end(), _comp);
		}
		void push(T&& value){
			_container.push_back(std::move(value));
			std::push_heap(_container.begin(), _container.end(), _comp);
		}
		T pop(){
			if(_container.empty())
				return {};
			T el = _container[0];
			std::pop_heap(_container.begin(), _container.end(), _comp);
			_container.pop_back();
			return el;
		}
	private:
		Container _container{};
		Compare _comp{};
	};
};

#endif