#include "PCUtil.h"
#include <cmath>
#include <netcdf.h>
#include <mutex>
#include <condition_variable>
#include "Attribute.hpp"

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

bool PCUtil::vectorEqual(const std::vector<float>& a, const std::vector<float>& b)
{
	if (a.size() != b.size()) return false;
	for (int i = 0; i < a.size(); ++i)
		if (a[i] != b[i])
			return false;
	return true;
}

float PCUtil::distance2(const ImVec2& a, const ImVec2& b){
	float x = a.x - b.x, y = a.y - b.y;
	return {x * x + y * y};
}

float PCUtil::distance(const ImVec2& a, const ImVec2& b){
	return std::sqrt(distance2(a, b));
}

bool PCUtil::compareStringFormat(const std::string_view& s, const std::string_view& form) 
{
	std::size_t curPos = 0, sPos = 0;
    while(true){
        std::size_t nextPos = form.find("*", curPos);
		if(nextPos == std::string_view::npos)
			break;
        std::string_view curPart = form.substr(curPos, nextPos - curPos);
		sPos = s.find(curPart, sPos);
        if(sPos == std::string_view::npos)
            return false;
        sPos += curPart.size();
        curPos = nextPos + 1;
    }
    return true;
}

std::vector<QueryAttribute> PCUtil::queryNetCDF(const std::string_view& filename) 
{
	int fileId, retval;
	if ((retval = nc_open(filename.data(), NC_NOWRITE, &fileId))) {
		std::cout << "Error opening the file" << std::endl;
		nc_close(fileId);
		return {};
	}

	int ndims, nvars, ngatts, unlimdimid;
	if ((retval = nc_inq(fileId, &ndims, &nvars, &ngatts, &unlimdimid))) {
		std::cout << "Error reading out viariable information" << std::endl;
		nc_close(fileId);
		return {};
	}

	std::vector<QueryAttribute> out;
	//getting all dimensions to distinguish the size for the data arrays
	uint32_t data_size = 0;
	std::vector<int> dimSizes(ndims);
	std::vector<bool> dimIsStringLenght(ndims);
	for (int i = 0; i < ndims; ++i) {
		size_t dim_size;
		if ((retval = nc_inq_dimlen(fileId, i, &dim_size))) {
			std::cout << "Error reading out dimension size" << std::endl;
			nc_close(fileId);
			return {};
		}
		//check for string length
		bool isStringLength = true;
		for(int j = 0; j < nvars; ++j){
			int dimensions;
			if((retval = nc_inq_varndims(fileId, j, &dimensions))){
				std::cout << "Error at getting variable dimensions(string length check)" << std::endl;
				nc_close(fileId);
				return {};
			}
			std::vector<int> dims(dimensions);
			if((retval = nc_inq_vardimid(fileId, j, dims.data()))){
				std::cout << "ERror at getting varialbe dimensions(string length check)" << std::endl;
				nc_close(fileId);
				return {};
			}
			//only check if current dimension is used
			if(std::find(dims.begin(), dims.end(), i) != dims.end()){
				nc_type varType;
				if((retval = nc_inq_vartype(fileId, j, &varType))){
					std::cout << "Error at getting variable type(string length check)" << std::endl;
					nc_close(fileId);
					return {};
				}
				if(varType != NC_CHAR){
					isStringLength = false;
					break;
				}
			}
		}
		if(!isStringLength){
			if (data_size == 0) data_size = dim_size;
			else data_size *= dim_size;
		}
		//else{
		//	std::cout << "Dim " << i << " is a stringlength dimension" << std::endl;
		//}
		dimSizes[i] = dim_size;
		dimIsStringLenght[i] = isStringLength;
	}
	//std::cout << "netCDF data size: " << data_size << std::endl;
	//for (int i = 0; i < data.size(); ++i) {
	//	data[i].resize(data_size);
	//}

	std::vector<std::vector<int>> attribute_dims;

	char vName[NC_MAX_NAME];
	for (int i = 0; i < nvars; ++i) {
		if ((retval = nc_inq_varname(fileId, i, vName))) {
			std::cout << "Error at reading variables" << std::endl;
			nc_close(fileId);
			return {};
		}
		out.push_back({ std::string(vName), 0, 1, -1, 0, 0, 0, true, false });
		if ((retval = nc_inq_varndims(fileId, i, &out.back().dimensionality))) {
			std::cout << "Error at getting variable dimensions" << std::endl;
			nc_close(fileId);
			return {};
		}
	}

	//creating the indices of the dimensions. Fastest varying is the last of the dimmensions
	std::vector<size_t> iter_indices(ndims), iter_stops(ndims);
	std::vector<int> dimension_variable_indices(ndims);
	for (int i = 0; i < ndims; ++i) {
		char dimName[NC_MAX_NAME];
		if ((retval = nc_inq_dim(fileId, i, dimName, &iter_stops[i]))) {
			std::cout << "Error at reading dimensions 2" << std::endl;
			nc_close(fileId);
			return {};
		}
		//dimensions are not attributes, they can however appear as an attribute too in the list
		//check if dimension is a string length that should not be shown
		if(dimIsStringLenght[i])
			out.push_back(QueryAttribute{ std::string(dimName), -dimSizes[i], 1, 1, 0, 0, dimSizes[i], true, false });
		else
			out.push_back(QueryAttribute{ std::string(dimName), dimSizes[i], 1, 1, 0, 0, dimSizes[i], true, false });
		//if ((retval = nc_inq_varid(fileId, dimName, &dimension_variable_indices[i]))) {
		//	// dimensions can exist as dimensions exclusiveley
		//	std::cout << "Error at getting variable id of dimension" << std::endl;
		//	nc_close(fileId);
		//	return {};
		//}
		//std:: cout << "Dimension " << dimName << " at index " << dimension_variable_indices[i] << " with lenght" << iter_stops[i] << std::endl;
	}
	//int c = 0;
	//for (int i : dimension_variable_indices) {
	//	out[i].dimensionSize = iter_stops[c++];
	//	out[i].trimIndices[1] = out[i].dimensionSize;
	//}

	//everything needed was red, so colosing the file
	nc_close(fileId);

	return out;
}

std::vector<int> PCUtil::checkAttributes(std::vector<std::string>& a, std::vector<Attribute>& ref) 
{
	if (ref.size() == 0) {
		std::vector<int> permutation;
		for (int i = 0; i < a.size(); i++) {
			permutation.push_back(i);
		}
		return permutation;
	}

	if (a.size() != ref.size())
		return std::vector<int>();

	//creating sets to compare the attributes
	std::set<std::string> pcAttr, attr;
	for (Attribute& a : ref) {
		std::string s = a.name;
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
		if (s == "rlon") s = "longitude";
		if (s == "rlat") s = "latitude";
		pcAttr.insert(s);
	}
	for (std::string s : a) {
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
		if (s == "rlon") s = "longitude";
		if (s == "rlat") s = "latitude";
		attr.insert(s);
	}

	if (pcAttr == attr) {
		//getting the right permutation
		std::vector<std::string> lowerCaseA(a), lowerCaseAttr;
		for (int i = 0; i < lowerCaseA.size(); i++) {
			std::transform(lowerCaseA[i].begin(), lowerCaseA[i].end(), lowerCaseA[i].begin(), [](unsigned char c) { return std::tolower(c); });
			if (lowerCaseA[i] == "rlon") lowerCaseA[i] = "longitude";
			if (lowerCaseA[i] == "rlat") lowerCaseA[i] = "latitude";
			std::string attribute = ref[i].name;
			std::transform(attribute.begin(), attribute.end(), attribute.begin(), [](unsigned char c) { return std::tolower(c); });
			if (attribute == "rlon") attribute = "longitude";
			if (attribute == "rlat") attribute = "latitude";
			lowerCaseAttr.push_back(attribute);
		}

		std::vector<int> permutation;
		for (std::string& s : lowerCaseA) {
			int i = 0;
			for (; i < lowerCaseAttr.size(); i++) {
				if (lowerCaseAttr[i] == s) break;
			}
			permutation.push_back(i);
		}

		return permutation;
	}

	return std::vector<int>();
}

Data PCUtil::openNetCdf(const std::string_view& filename,  /*inout*/ std::vector<Attribute>& inoutAttributes, const std::vector<QueryAttribute>& queryAttributes) 
{
	int fileId, retval;
    if((retval = nc_open(filename.data(), NC_NOWRITE,&fileId))){
        std::cout << "Error opening the file" << std::endl;
        nc_close(fileId);
        return {};
    }
    
    int ndims, nvars, ngatts, unlimdimid;
    if((retval = nc_inq(fileId, &ndims, &nvars, &ngatts, &unlimdimid))){
        std::cout << "Error at reading out viariable information" << std::endl;
        nc_close(fileId);
        return {};
    }
    
    //attribute check
    std::vector<Attribute> tmp;
    std::vector<std::string> attributes;
    std::vector<std::vector<int>> attribute_dims;
    std::vector<std::vector<int>> variable_dims;
	std::vector<int> attr_to_var;
    
    char vName[NC_MAX_NAME];
    for(int i = 0; i < nvars; ++i){
        if((retval = nc_inq_varname(fileId, i, vName))){
            std::cout << "Error at reading variables" << std::endl;
            nc_close(fileId);
            return {};
        }
		int ndims;
		if ((retval = nc_inq_varndims(fileId, i, &ndims))) {
			std::cout << "Error at getting variable dimensions" << std::endl;
			nc_close(fileId);
			return {};
		}
		variable_dims.push_back(std::vector<int>(ndims));
		if ((retval = nc_inq_vardimid(fileId, i, variable_dims.back().data()))) {
			std::cout << "Error at getting variable dimension array" << std::endl;
			nc_close(fileId);
			return {};
		}
		if (queryAttributes[i].active) {
			tmp.push_back({ vName, vName,{},{},std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() });
			attributes.push_back(tmp.back().name);
			attr_to_var.push_back(i);
			attribute_dims.push_back(variable_dims[i]);
		}
        //std::cout << vName << "(";
        //for(int dim: attribute_dims.back()){
        //    std::cout << dim << ", ";
        //}
        //std::cout << "\b\b)" << std::endl;
    }
    
    //creating the indices of the dimensions. Fastest varying is the last of the dimmensions
    std::vector<size_t> iter_indices(ndims), iter_stops(ndims), iter_increments(ndims, 1), iter_starts(ndims, 0);
	std::vector<size_t> dim_sizes(ndims);
    //std::vector<int> dimension_variable_indices(ndims);
	std::vector<bool> dimension_is_stringsize(ndims);
    for(int i = 0; i < ndims; ++i){
        char dimName[NC_MAX_NAME];
        if((retval = nc_inq_dim(fileId, i, dimName, &dim_sizes[i]))){
            std::cout << "Error at reading dimensions 2" << std::endl;
            nc_close(fileId);
            return {};
        }
        //if((retval = nc_inq_varid(fileId, dimName, &dimension_variable_indices[i]))){
        //    std::cout << "Error at getting variable id of dimension" << std::endl;
        //    nc_close(fileId);
        //    return false;
        //}
		dimension_is_stringsize[i] = queryAttributes[queryAttributes.size() - ndims + i].dimensionSize < 0;
		iter_increments[i] = queryAttributes[queryAttributes.size() - ndims + i].dimensionSubsample;
		iter_starts[i] = queryAttributes[queryAttributes.size() - ndims + i].trimIndices[0];
		iter_stops[i] = queryAttributes[queryAttributes.size() - ndims + i].trimIndices[1];
		iter_indices[i] = iter_starts[i];
		if (!queryAttributes[queryAttributes.size() - ndims + i].active) {
			iter_indices[i] = queryAttributes[queryAttributes.size() - ndims + i].dimensionSlice;
			iter_starts[i] = iter_indices[i];
			iter_stops[i] = iter_indices[i] + 1;
		}
        //std:: cout << "Dimension " << dimName << " at index " << dimension_variable_indices[i] << " with lenght" << iter_stops[i] << std::endl;
    }

	std::vector<float> fill_values(nvars);
	std::vector<float> has_fill_value(nvars);
	std::vector<std::vector<std::string>> categories(nvars);
	//getting all dimensions to distinguish the size for the data arrays
	uint32_t data_size = 0;
	uint32_t reduced_data_size = 1;
	for (int i = 0; i < ndims; ++i) {
		size_t dim_size;
		if ((retval = nc_inq_dimlen(fileId, i, &dim_size))) {
			std::cout << "Error at reading out dimension size" << std::endl;
			nc_close(fileId);
			return {};
		}
		if (data_size == 0) data_size = dim_size;
		else data_size *= dim_size;
		if (!dimension_is_stringsize[i]) {
			dim_size = queryAttributes[queryAttributes.size() - ndims + i].trimIndices[1] - queryAttributes[queryAttributes.size() - ndims + i].trimIndices[0];
			dim_size = dim_size / queryAttributes[queryAttributes.size() - ndims + i].dimensionSubsample + ((dim_size % queryAttributes[queryAttributes.size() - ndims + i].dimensionSubsample) ? 1 : 0);
			reduced_data_size *= dim_size;
		}
	}

	//attribute check
	//checking if the Attributes are correct
	std::vector<int> permutation = checkAttributes(attributes, inoutAttributes);
    if (inoutAttributes.size() != 0) {
        if (tmp.size() != inoutAttributes.size()) {
            std::cout << "The Amount of Attributes of the .nc file is not compatible with the currently loaded datasets" << std::endl;
            return {};
        }

        if (!permutation.size()) {
            std::cout << "The attributes of the .nc data are not the same as the ones already loaded in the program." << std::endl;
            return {};
        }
	}
	//if this is the first Dataset to be loaded, fill the pcAttributes vector
	else {
		inoutAttributes = tmp;

		//setting up the categorical datastruct
		for(int i = 0; i < inoutAttributes.size(); ++i){
			if(categories[attr_to_var[i]].size()){
				//we do have categorical data
				std::vector<std::pair<std::string,int>> lexicon;
				int c = 0;
				for (auto& categorie : categories[attr_to_var[i]]) {
					lexicon.push_back({ categorie, c });
					c++;
				}
				std::sort(lexicon.begin(), lexicon.end(), [](auto& a, auto& b) {return a.first < b.first; });	//after sorting the names are ordered in lexigraphical order, the seconds are the original indices
		
				for(int j = 0; j < categories[attr_to_var[i]].size() ; ++j){
					inoutAttributes[i].categories[lexicon[j].first] = j;
					inoutAttributes[i].categories_ordered.push_back({lexicon[j].first, j});
				}
			}
		}
	}

	//preparing the data member of dataset to hold the data
	Data ds{};
	ds.dimensionSizes = std::vector<uint32_t>(dim_sizes.begin(), dim_sizes.end());
	ds.columnDimensions.resize(attribute_dims.size());
	ds.columns.resize(tmp.size());
	ds.columnDimensions.resize(tmp.size());
	for(int i = 0; i < ds.columns.size(); ++i){
		int var = attr_to_var[permutation[i]];
		ds.columnDimensions[permutation[i]] = std::vector<uint32_t>(attribute_dims[i].begin(), attribute_dims[i].end());
		int columnSize = 1;
		for (int dim : attribute_dims[i]) {
			columnSize *= ds.dimensionSizes[dim];
		}
		ds.columns[permutation[i]].resize(columnSize);
	}
	for(int i = 0; i < dim_sizes.size(); ++i){
		if(dimension_is_stringsize[i])
			ds.removeDim(i, 0);
	}

	//reading out all data from the netCDF file(including conversion)
	for (int i = 0; i < ds.columns.size(); ++i) {
		int var = attr_to_var[permutation[i]];
		nc_type type;
		if ((retval = nc_inq_vartype(fileId, var, &type))){
			std::cout << "Error at reading data type" << std::endl;
			nc_close(fileId);
			return {};
		}
		int hasFill = 0; // if 1 no fill
		float fillValue = 0;
		switch(type){
			case NC_FLOAT:
			if ((retval = nc_get_var_float(fileId, var, ds.columns[i].data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return {};
			}
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &fillValue))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return {};
			}
			break;
			case NC_DOUBLE:
			{
			auto d = std::vector<double>(ds.columns[i].size());
			if ((retval = nc_get_var_double(fileId, var, d.data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return {};
			}
			ds.columns[i] = std::vector<float>(d.begin(), d.end());
			double f = 0;
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return {};
			}
			fillValue = f;
			break;
			}
			case NC_INT:
			{
			auto d = std::vector<int>(ds.columns[i].size());
			if ((retval = nc_get_var_int(fileId, var, d.data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return {};
			}
			ds.columns[i] = std::vector<float>(d.begin(), d.end());
			int f = 0;
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return {};
			}
			fillValue = f;
			break;
			}
			case NC_UINT:
			{
			auto d = std::vector<uint32_t>(ds.columns[i].size());
			if ((retval = nc_get_var_uint(fileId, var, d.data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return {};
			}
			ds.columns[i] = std::vector<float>(d.begin(), d.end());
			uint32_t f = 0;
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return {};
			}
			fillValue = f;
			break;
			}
			case NC_UINT64:{
				auto d = std::vector<unsigned long long>(ds.columns[i].size());
				if((retval = nc_get_var_ulonglong(fileId, var, d.data()))){
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return {};
				}
				ds.columns[i] = std::vector<float>(d.begin(), d.end());
				unsigned long long f = 0;
				if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return {};
				}
				fillValue = f;
				break;
			}
			case NC_INT64:{
				auto d = std::vector<long long>(ds.columns[i].size());
				if((retval = nc_get_var_longlong(fileId, var, d.data()))){
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return {};
				}
				ds.columns[i] = std::vector<float>(d.begin(), d.end());
				long long f = 0;
				if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return {};
				}
				fillValue = f;
				break;
			}
			case NC_UBYTE:
			case NC_BYTE:{
				auto d = std::vector<uint8_t>(ds.columns[i].size());
				if((retval = nc_get_var_ubyte(fileId, var, d.data()))){
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return {};
				}
				ds.columns[i] = std::vector<float>(d.begin(), d.end());
				uint8_t f = 0;
				if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return {};
				}
				fillValue = f;
				break;
			}
			case NC_CHAR:
			//categorical data
			{
				int dataSize = 1;
				int amtOfDims;
				if((retval = nc_inq_varndims(fileId, var, &amtOfDims))){
					std::cout << "Error at reading dimsizes for categorical data" << std::endl;
					nc_close(fileId);
					return {};
				}
				std::vector<int> dims(amtOfDims);
				if((retval = nc_inq_vardimid(fileId, var, dims.data()))){
					std::cout << "Error at reading dims for categorical data" << std::endl;
					nc_close(fileId);
					return {};
				}
				int wordlen = 0;
				for(auto dim: dims){
					dataSize *= std::abs(queryAttributes[queryAttributes.size() - ndims + dim].dimensionSize);
					if(queryAttributes[queryAttributes.size() - ndims + dim].dimensionSize < 0){
						wordlen = -queryAttributes[queryAttributes.size() - ndims + dim].dimensionSize;
					}
				}
				std::vector<char> names(dataSize);
				if((retval = nc_get_var_text(fileId, i, names.data()))){
					std::cout << "Error at reading categorical data" << std::endl;
					nc_close(fileId);
					return {};
				}
				int c = 0;
				for(int offset = 0; offset < dataSize; offset += wordlen){
					categories[i].push_back(std::string(&names[offset], &names[offset] + wordlen));
					inoutAttributes[i].categories[categories[i].back()] = c++;
					inoutAttributes[i].categories_ordered.push_back({categories[i].back(), float(inoutAttributes[i].categories[categories[i].back()])});
					ds.columns[i][offset / wordlen] = inoutAttributes[i].categories[categories[i].back()];
				}
				std::sort(inoutAttributes[i].categories_ordered.begin(), inoutAttributes[i].categories_ordered.end(), [&](auto& left, auto& right){return left.second < right.second;});
			}
			break;
			default:
				std::cout << "The variable type " << type << " can not be handled correctly!" << std::endl;
		}
		if (hasFill != 1) {
			fill_values[i] = fillValue;
			has_fill_value[i] = true;
		}
	}
    
    //everything needed was red, so colosing the file
    nc_close(fileId);

	for (int j = 0; j < inoutAttributes.size(); j++) {
		for (float f: ds.columns[j]) {
			//ignoring fill values
			if (has_fill_value[j] && f == fill_values[j] && inoutAttributes[j].categories.empty())
				continue;
            //updating pcAttributes minmax if needed
            if(f < inoutAttributes[j].min)
                inoutAttributes[j].min = f;
            if(f > inoutAttributes[j].max)
                inoutAttributes[j].max = f;
		}
	}
	return ds;
}

PCUtil::Stopwatch::Stopwatch(std::ostream& stream, const std::string& displayName):
_ostream(stream),
_name(displayName)
{
	_start = std::chrono::high_resolution_clock::now();
}

PCUtil::Stopwatch::~Stopwatch(){
	auto end = std::chrono::high_resolution_clock::now();
	_ostream << "Stopwatch " << _name << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - _start).count() << " ms" << std::endl;
}

PCUtil::AverageWatch::AverageWatch(float& average, uint32_t& count) :
_a(average),
_c(count)
{
	_start = std::chrono::high_resolution_clock::now();
}

PCUtil::AverageWatch::~AverageWatch() 
{
	auto end = std::chrono::high_resolution_clock::now();
	float t = _c / float(++_c);
	_a = t * _a + (1 - t) * std::chrono::duration_cast<std::chrono::nanoseconds>(end - _start).count() * 1e-3;
}

PCUtil::AverageWatch& PCUtil::AverageWatch::operator=(const PCUtil::AverageWatch & o) 
{
	this->_a = o._a;
	this->_c = o._c;
	this->_start = o._start;
	return *this;
}

void PCUtil::Semaphore::release() 
{
	std::lock_guard<decltype(_mutex)> lock(_mutex);
	++_count;
	_cv.notify_one();
}

void PCUtil::Semaphore::releaseN(int n) 
{
	std::lock_guard<decltype(_mutex)> lock(_mutex);
	_count = n;
	_cv.notify_all();
}

void PCUtil::Semaphore::acquire() 
{
	std::unique_lock<decltype(_mutex)> lock(_mutex);
	while(!_count)
		_cv.wait(lock);
	--_count;
}