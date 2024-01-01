#ifndef Cylinder_H
#define Cylinder_H

#include <vector>
#include "glm/glm/glm.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Shape.h"

class Cylinder: public Shape {
public:
    //creates a clinder of with radius r and length l centered at the origin. Res specifies the amount of segments used for the bottom and top circle
    Cylinder(float r, float length, int res) : Shape() {
        //filling the vertex buffer
        vertexBuffer.reserve(2 * res);
        //lower ring
        for (int i = 0; i < res; i++) {
            Vertex cur;
            cur.position = glm::vec3( r * sin(i * 2 * M_PI / res),-length / 2,r * cos(i * 2 * M_PI / res) );
            cur.normal = cur.position;
            cur.normal.y = 0;
            cur.normal = glm::normalize(cur.normal);
            vertexBuffer.push_back(cur);
        }
        //upper ring
        for (int i = 0; i < res; i++) {
            Vertex cur;
            cur.position = { r * sin(i * 2 * M_PI / res),length / 2,r * cos(i * 2 * M_PI / res) };
            cur.normal = cur.position;
            cur.normal.y = 0;
            cur.normal = glm::normalize(cur.normal);
            vertexBuffer.push_back(cur);
        }

        //filling the indexbuffer
        for (int i = 0; i < res; i++) {
            indexBuffer.push_back(i);
            indexBuffer.push_back(i +res);
            indexBuffer.push_back((i + 1) % res + res);

            indexBuffer.push_back((i + 1) % res + res);
            indexBuffer.push_back((i + 1) % res);
            indexBuffer.push_back(i);
        }
    };

    ~Cylinder() {};

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