#ifndef PCUtil_H
#def PCUtil_H

#include <istream>
#include <vector>

class PCUtil {
public:
	static std::vector<char> readByteFile(const std::string& filename);
};

#endif