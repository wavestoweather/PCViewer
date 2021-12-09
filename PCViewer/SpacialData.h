#ifndef SpacialData_H
#define SpacialData_H

#include <vector>
#include <math.h>

class SpacialData {
public:
	static float altitude[];
	static float rlon[];
	static float rlat[];
	static int altitudeSize;
	static int rlonSize;
	static int rlatSize;
 
	static int getR(float a, float* arr, int start, int end);

	static int getAltitudeIndex(float a);

	static int getRlonIndex(float a);

	static int getRlatIndex(float a);
};

#endif // !SpacialData_H
