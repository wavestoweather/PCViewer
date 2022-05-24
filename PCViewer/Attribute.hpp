#pragma once
#include <map>
#include <string>
#include <vector>

struct Attribute {
	std::string name;
	std::string originalName;
	std::map<std::string, float> categories;	//if data has a categorical structure the categories map will be filled.
	std::vector<std::pair<std::string, float>> categories_ordered; // used to show the categories not cluttered
	float min;			//min value of all values
	float max;			//max value of all values
	bool operator==(const Attribute& other) const{
		return name == other.name;
	}
};

struct QueryAttribute {
	std::string name;
	int dimensionSize;	//size of the dimension, 0 if not a dimension, negative if stringlength dimension
	int dimensionality; //amt of dimensions the attribute is dependant on
	int dimensionSubsample; //sampling rate of the dimension to reduce its size
	int dimensionSlice; //if < 0 the whole dimension should be taken, otherwise the dimension is disabled and sliced at the index indicated here
	int trimIndices[2];
	bool active;
	bool linearize;

	bool operator==(const QueryAttribute& other) const {
		return name == other.name && dimensionSize == other.dimensionSize && dimensionality == other.dimensionality;
	}
};