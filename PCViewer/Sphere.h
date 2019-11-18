#ifndef Sphere_H
#define Sphere_H

#include <vector>
#include "glm/glm/glm.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Shape.h"
#include <stdint.h>

class Sphere: public Shape {
public:
	//creates a sphere of radius r around origin. The res parameter specifies, how much segments the sphere will be split into
	Sphere(float r, int res) : Shape() {
		//filling the vertex Buffer
		float xy,z;
		float sectorStep = 2 * M_PI / res;
		float stackStep = M_PI / res;
		float sectorAngle, stackAngle, lengthInv = 1.0f / r;;

		for (int i = 0; i <= res; ++i)
		{
			stackAngle = M_PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
			xy = r * cosf(stackAngle);             // r * cos(u)
			z = r * sinf(stackAngle);              // r * sin(u)

			// add (sectorCount+1) vertices per stack
			// the first and last vertices have same position and normal, but different tex coords
			for (int j = 0; j <= res; ++j)
			{
				sectorAngle = j * sectorStep;           // starting from 0 to 2pi

				// vertex position (x, y, z)
				Vertex cur;
				cur.position.x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
				cur.position.y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
				cur.position.z = z;

				// normalized vertex normal (nx, ny, nz)
				cur.normal = cur.position * lengthInv;

				vertexBuffer.push_back(cur);
			}
		}

		//filling the index buffer
		int k1, k2;
		for (int i = 0; i < res; ++i)
		{
			k1 = i * (res + 1);     // beginning of current stack
			k2 = k1 + res + 1;      // beginning of next stack

			for (int j = 0; j < res; ++j, ++k1, ++k2)
			{
				// 2 triangles per sector excluding first and last stacks
				// k1 => k2 => k1+1
				if (i != 0)
				{
					indexBuffer.push_back(k1);
					indexBuffer.push_back(k2);
					indexBuffer.push_back(k1 + 1);
				}

				// k1+1 => k2 => k2+1
				if (i != (res - 1))
				{
					indexBuffer.push_back(k1 + 1);
					indexBuffer.push_back(k2);
					indexBuffer.push_back(k2 + 1);
				}
			}
		}

		//vertexBuffer.reserve(res * (res - 1) / 2 + 2);
		//vertexBuffer.push_back({ glm::vec3(0,r,0),glm::vec3(0,1,0) });
		//for (int i = 1; i < res / 2; i++) {
		//	for (int j = 0; j < res; j++) {
		//		Vertex cur;
		//		cur.position = glm::vec3(r * sin(i * 2 * M_PI / res) * cos(j * 2 * M_PI / res), r * cos(i * 2 * M_PI / res), r * sin(i * 2 * M_PI / res) * sin(j * 2 * M_PI / res));
		//		cur.normal = glm::normalize(cur.position);
		//		vertexBuffer.push_back(cur);
		//	}
		//}
		//vertexBuffer.push_back({ glm::vec3(0,-r,0),glm::vec3(0,-1,0) });
		//
		////filling the index Buffer
		//for (int i = 0; i < res / 2 - 1; i++) {
		//	//special cases
		//	if (i == 0) {
		//		for (int j = 0; j < res; j++) {
		//			indexBuffer.push_back(0);
		//			indexBuffer.push_back(j + 1);
		//			indexBuffer.push_back(((j + 1) % res) + 1);
		//		}
		//	}
		//	else if (i == res / 2 - 2) {
		//		for (int j = 0; j < res; j++) {
		//			indexBuffer.push_back((i + 1) * res + 1);
		//			indexBuffer.push_back((i)* res + (j + 1) % res + 1);
		//			indexBuffer.push_back((i)* res + j + 1);
		//		}
		//	}
		//	else {
		//		for (int j = 0; j < res; j++) {
		//			indexBuffer.push_back(i * res + j + 1);
		//			indexBuffer.push_back((i + 1) * res + j + 1);
		//			indexBuffer.push_back((i + 1) * res + ((j + 1) % res) + 1);
		//
		//			indexBuffer.push_back((i + 1) * res + ((j + 1) % res) + 1);
		//			indexBuffer.push_back(i * res * ((j + 1) % res) + 1);
		//			indexBuffer.push_back(i * res + j + 1);
		//		}
		//	}
		//}
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