#ifndef Shape_H
#define Shape_H

#include <vector>

class Shape {
public:
	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
	};

	Shape() {};
	~Shape() {};
};

#endif