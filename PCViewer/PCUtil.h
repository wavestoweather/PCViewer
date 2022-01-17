#ifndef PCUtil_H
#define PCUtil_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include "imgui/imgui.h"

class PCUtil {
public:
	static std::vector<char> readByteFile(const std::string& filename);
	static void hexdump(const void* ptr, int buflen);
	static void numdump(const float* ptr, int len);
	static void numdump(const int* ptr, int len);
	static void matrixdump(const std::vector<std::vector<int>>& matrix);
	static void matrixdump(const std::vector<std::vector<double>>& matrix);
	static float getVectorIndex(const std::vector<float>& values, float v);
	static bool vectorEqual(const std::vector<float>& a, const std::vector<float>& b);
	static float distance2(const ImVec2& a, const ImVec2& b);
	static float distance(const ImVec2& a, const ImVec2& b);

	class Stopwatch{
		public:
		Stopwatch(std::ostream& stream, const std::string& displayName);
		~Stopwatch();
		private:
		std::ostream& _ostream;
		std::string _name;
		std::chrono::high_resolution_clock::time_point _start;
	};
};

#endif