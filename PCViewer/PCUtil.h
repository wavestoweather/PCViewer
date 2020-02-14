#ifndef PCUtil_H
#define PCUtil_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

class PCUtil {
public:
	static std::vector<char> readByteFile(const std::string& filename);
	static void hexdump(void* ptr, int buflen);
};

#endif