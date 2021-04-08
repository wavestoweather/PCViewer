#include "PCUtil.h"

std::vector<char> PCUtil::readByteFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "failed to open file " << filename << "!" << std::endl;
		exit(-1);
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

void PCUtil::hexdump(const void* ptr, int buflen) {
	unsigned char* buf = (unsigned char*)ptr;
	int i, j;
	for (i = 0; i < buflen; i += 16) {
		printf("%06x: ", i);
		for (j = 0; j < 16; j++)
			if (i + j < buflen)
				printf("%02x ", buf[i + j]);
			else
				printf("   ");
		printf(" ");
		for (j = 0; j < 16; j++)
			if (i + j < buflen)
				printf("%c", isprint(buf[i + j]) ? buf[i + j] : '.');
		printf("\n");
	}
}

void PCUtil::numdump(const float* ptr, int len)
{
	for (int i = 0; i < len; ++i) {
		std::cout << ptr[i] << std::endl;
	}
}

void PCUtil::numdump(const int* ptr, int len)
{
	for (int i = 0; i < len; ++i) {
		std::cout << ptr[i] << std::endl;
	}
}

void PCUtil::matrixdump(const std::vector<std::vector<int>>& matrix)
{
	for (auto& x : matrix) {
		for (auto& y : x) {
			std::cout << y << " ";
		}
		std::cout << std::endl;
	}
}

void PCUtil::matrixdump(const std::vector<std::vector<double>>& matrix)
{
	for (auto& x : matrix) {
		for (auto& y : x) {
			std::cout << y << " ";
		}
		std::cout << std::endl;
	}
}

float PCUtil::getVectorIndex(const std::vector<float>& values, float v)
{
	//binary search
	int a = 0, b = values.size();
	while (b - a > 1) {
		int half = (b + a) / 2;
		float val = values[half];
		if (v < val)
			b = half;
		else
			a = half;
	}
	//a now at begin index, b at endindex
	if (a == values.size() - 1)
		return a;
	float af = values[a], bf = values[b];
	return (v - af) / (bf - af) + a;
}
