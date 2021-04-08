#ifndef PCUtil_H
#define PCUtil_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

class PCUtil {
public:
	static std::vector<char> readByteFile(const std::string& filename);
	static void hexdump(const void* ptr, int buflen);
	static void numdump(const float* ptr, int len);
	static void numdump(const int* ptr, int len);
	static void matrixdump(const std::vector<std::vector<int>>& matrix);
	static void matrixdump(const std::vector<std::vector<double>>& matrix);
	static float getVectorIndex(const std::vector<float>& values, float v);
};

#endif