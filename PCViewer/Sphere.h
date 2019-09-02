#ifndef Sphere_H
#define Sphere_H

#include <vector>
#include "glm/glm/glm.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Shape.h"

class Sphere: public Shape {
public:
	//creates a sphere of radius r around origin. The res parameter specifies, how much segments the sphere will be split into
	Sphere(float r, int res) : Shape() {
		//filling the vertex Buffer
		vertexBuffer.reserve(res * (res - 1) / 2 + 2);
		vertexBuffer.push_back({ glm::vec3(0,-r,0),glm::vec3(0,-1,0) });
		for (int i = 1; i < res / 2; i++) {
			for (int j = 0; j < res; j++) {
				Vertex cur;
				cur.position = glm::vec3(r * sin(i * 2 * M_PI / res) * cos(j * 2 * M_PI / res), r * cos(i * 2 * M_PI / res), r * sin(i * 2 * M_PI / res) * sin(j * 2 * M_PI / res));
				cur.normal = glm::normalize(cur.position);
				vertexBuffer.push_back(cur);
			}
		}
		vertexBuffer.push_back({ glm::vec3(0,r,0),glm::vec3(0,1,0) });

		//filling the index Buffer
		for (int i = 0; i < res / 2; i++) {
			//special cases
			if (i == 0) {
				for (int j = 0; j < res; j++) {
					indexBuffer.push_back(0);
					indexBuffer.push_back(i);
					indexBuffer.push_back((i + 1) % res);
				}
			}
			else if (i == res / 2 - 1) {
				for (int j = 0; j < res; j++) {
					indexBuffer.push_back((i + 1) * res + 1);
					indexBuffer.push_back((i)* res + (j + 1) % res + 1);
					indexBuffer.push_back((i)* res + j + 1);
				}
			}
			else {
				for (int j = 0; j < res; j++) {
					indexBuffer.push_back(i * res + j + 1);
					indexBuffer.push_back((i + 1) * res + j + 1);
					indexBuffer.push_back((i + 1) * res + (j + 1) % res + 1);

					indexBuffer.push_back((i + 1) * res + (j + 1) % res + 1);
					indexBuffer.push_back(i * res * (j + 1) % res + 1);
					indexBuffer.push_back(i * res + j + 1);
				}
			}
		}
	};

	~Sphere() {};

	std::vector<Vertex> getVertexBuffer() {
		return std::vector<Vertex>(vertexBuffer);
	};

	std::vector<uint32_t> getIndexBuffer(uint32_t startOffset) {
		std::vector<uint32_t> iB(indexBuffer);

		if (startOffset == 0)
			return iB;

		for (uint32_t& i : iB) {
			i += startOffset;
		}

		return iB;
	};

private:
	std::vector<Vertex> vertexBuffer;
	std::vector<uint32_t> indexBuffer;
};

#endif