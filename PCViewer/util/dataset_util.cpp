#include "dataset_util.hpp"
#include <netcdf_file.hpp>
#include <c_file.hpp>
#include <sstream>
#include <filesystem>
#include <charconv>
#include <robin_hood.h>
#include <functional>
#include <set>

namespace util{
namespace dataset{
namespace open_internals{

load_result<float> open_netcdf_float(std::string_view filename, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
    structures::netcdf_file netcdf(filename);
	auto& variables = netcdf.get_variable_infos();
	if(query_attributes.size()){
		for(int var: util::size_range(variables)){
			if(query_attributes[var].id != variables[var].name)
				throw std::runtime_error{"open_netcdf() Attributes of the attribute query are not consistent with the netcdf file"};
		}
	}
	load_result<float> ret{};
	for(const auto& d: netcdf.get_dimension_infos())
		ret.data.dimension_sizes.push_back(d.size);
	
	for(int var: util::size_range(variables)){
		if(query_attributes.size() > var && !query_attributes[var].is_active)
			continue;
		ret.data.column_dimensions.push_back(std::vector<uint32_t>(variables[var].dependant_dimensions.begin(), variables[var].dependant_dimensions.end()));
		auto [data, fill_value] = netcdf.read_variable<float>(var);
		ret.data.columns.push_back(std::move(data));
		ret.fill_values.push_back(fill_value);
		ret.attributes.push_back(structures::attribute{variables[var].name, variables[var].name});
		for(float f: ret.data.columns.back()){
			if(ret.attributes.back().bounds.read().min > f)
				ret.attributes.back().bounds().min = f;
			if(ret.attributes.back().bounds.read().max < f)
				ret.attributes.back().bounds().max = f;
		}
		ret.attributes.back().data_bounds = ret.attributes.back().bounds.read();
	}

	return std::move(ret);
}

load_result<half> open_netcdf_half(std::string_view filename, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
    structures::netcdf_file netcdf(filename);
	auto variables = netcdf.get_variable_infos();
	if(query_attributes.size()){
		for(int var: util::size_range(variables)){
			if(query_attributes[var].id != variables[var].name)
				throw std::runtime_error{"open_netcdf() Attributes of the attribute query are not consistent with the netcdf file"};
		}
	}
	load_result<half> ret{};
	for(const auto& d: netcdf.get_dimension_infos())
		ret.data.dimension_sizes.push_back(d.size);
	
	for(int var: util::size_range(variables)){
		if(query_attributes.size() > var && !query_attributes[var].is_active)
			continue;
		ret.data.column_dimensions.push_back(std::vector<uint32_t>(variables[var].dependant_dimensions.begin(), variables[var].dependant_dimensions.end()));
		auto [data, fill_value] = netcdf.read_variable<half>(var);
		ret.data.columns.push_back(std::move(data));
		ret.fill_values.push_back(fill_value);
		ret.attributes.push_back(structures::attribute{variables[var].name, variables[var].name});
		for(float f: ret.data.columns.back()){
			if(ret.attributes.back().bounds.read().min > f)
				ret.attributes.back().bounds().min = f;
			if(ret.attributes.back().bounds.read().max < f)
				ret.attributes.back().bounds().max = f;
		}
		ret.attributes.back().data_bounds = ret.attributes.back().bounds.read();
	}

	return std::move(ret);
}

bool getline(std::string_view& input, std::string_view& element, char delimiter = '\n'){
	if(input.empty())
		return false;
	
	size_t delimiter_pos = input.find(delimiter);
	size_t start = delimiter_pos + 1;
	if(delimiter_pos == std::string_view::npos){
		delimiter_pos = input.size();
		start = delimiter_pos;
	}
	element = input.substr(0, delimiter_pos);
	input = input.substr(start, input.size() - start);
	return true;
}

void trim_inplace(std::string_view& str){
	str = str.substr(str.find_first_not_of(" "));
	size_t back = str.size() - 1;
	while(str[back] == ' ')
		--back;
	str = str.substr(0, back + 1);
}

std::string_view trim(const std::string_view& str){
	std::string_view v = str;
	trim_inplace(v);
	return v;
}

template<typename T, typename predicate>
void erase_if(std::vector<T>& c, predicate pred){
	auto it = std::remove_if(c.begin(), c.end(), pred);
	c.erase(it, c.end());
}

load_result<float> open_csv_float(std::string_view filename, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
	using T = float;
	
	std::string input;
	{
		size_t file_size = std::filesystem::file_size(filename);
		input.resize(file_size);
		structures::c_file csv(filename, "rb");
		assert(csv);
		csv.read(util::memory_view<char>{input.data(), input.size()});
	}
	std::string_view input_view(input);

	const char delimiter = ',';
	std::vector<std::string> variable_names;
	// reading header(including attribute checks)
	{
		std::string_view line; getline(input_view, line);
		for(std::string_view variable; getline(line, variable, ',');){
			variable_names.push_back(std::string(variable));
		}
		if(query_attributes.size()){
			for(int var: util::size_range(variable_names)){
				if(query_attributes.size() <= var || variable_names[var] != query_attributes[var].id)
					throw std::runtime_error{"open_csv() Attributes of the attribute query are not consistent with the csv file"};
			}
		}
	}

	// parsing the data
	load_result<T> ret{};
	std::map<uint32_t, T> category_values;
	ret.data.columns.resize(variable_names.size());
	ret.attributes.resize(variable_names.size());
	for(std::string_view line; getline(input_view, line);){
		int var = 0;
		for(std::string_view element; getline(line, element, ','); ++var){
			trim_inplace(element);
			// TODO quotation marks
			if(query_attributes.size() && !query_attributes[var].is_active)
				continue;
			if(var >= variable_names.size())
				throw std::runtime_error{"open_csv() Too much values for data row"};
			
			T val{};
			if(element.size()){
				auto parse_res = std::from_chars(element.begin(), element.end(), val);
				if(parse_res.ec != std::errc{}){	// parsing error -> exchnage for category
					std::string el(element);
					if(ret.attributes[var].categories.count(el) > 0)
						val = ret.attributes[var].categories[el];
					else{
						val = category_values[var];
						category_values[var] += 1;
						ret.attributes[var].categories[el] = val;
					}
				}
			}

			if(val > ret.attributes[var].bounds.read().max)
				ret.attributes[var].bounds().max = val;
			if(val < ret.attributes[var].bounds.read().min)
				ret.attributes[var].bounds().min = val;
			ret.data.columns[var].push_back(val);
		}
	}
	// lexicographically ordering categorical data (using the automatical sorting provided by map)
	for(int var: util::size_range(ret.attributes)){
		auto& categories = ret.attributes[var].categories;
		if(categories.empty())
			continue;
		std::map<T, T> category_conversion;
		uint32_t counter{};
		for(auto& [category, value]: categories){
			T val = counter++;
			category_conversion[value] = val;
			value = val;
		}
		for(T& f: ret.data.columns[var])
			f = category_conversion[f];
		
	}

	for(int var: util::size_range(ret.attributes)){
		ret.attributes[var].id = variable_names[var];
		ret.attributes[var].display_name = variable_names[var];
		ret.attributes[var].data_bounds = ret.attributes[var].bounds.read();
	}
	if(query_attributes.size()){
		std::set<std::string_view> active_attributes;
		for(const auto& a: query_attributes)
			if(a.is_active)
				active_attributes.insert(a.id);
		erase_if(ret.attributes, [&](const structures::attribute& att){return active_attributes.count(att.id) == 0;});
	}
	erase_if(ret.data.columns, [](const std::vector<T>& v){return v.empty();});
	ret.data.dimension_sizes = {static_cast<uint32_t>(ret.data.columns[0].size())};
	ret.data.column_dimensions = std::vector<std::vector<uint32_t>>(ret.data.columns.size(), {0});
	if(query_attributes.size())
		ret.data.subsampleTrim({static_cast<uint32_t>(query_attributes.back().dimension_subsample)}, {{0, ret.data.dimension_sizes[0]}});
	ret.data.compress();
	
	return std::move(ret);
}

load_result<half> open_csv_half(std::string_view filename, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
    using T = half;
	
	std::string input;
	{
		size_t file_size = std::filesystem::file_size(filename);
		input.resize(file_size);
		structures::c_file csv(filename, "rb");
		assert(csv);
		csv.read(util::memory_view<char>{input.data(), input.size()});
	}
	std::string_view input_view(input);

	const char delimiter = ',';
	std::vector<std::string> variable_names;
	// reading header(including attribute checks)
	{
		std::string_view line; getline(input_view, line);
		for(std::string_view variable; getline(line, variable, ',');){
			variable_names.push_back(std::string(variable));
		}
		if(query_attributes.size()){
			for(int var: util::size_range(variable_names)){
				if(query_attributes.size() <= var || variable_names[var] != query_attributes[var].id)
					throw std::runtime_error{"open_csv() Attributes of the attribute query are not consistent with the csv file"};
			}
		}
	}

	// parsing the data
	load_result<T> ret{};
	std::map<uint32_t, T> category_values;
	ret.data.columns.resize(variable_names.size());
	ret.attributes.resize(variable_names.size());
	for(std::string_view line; getline(input_view, line);){
		int var = 0;
		for(std::string_view element; getline(line, element); ++var){
			trim_inplace(element);
			// TODO quotation marks
			if(query_attributes.size() && !query_attributes[var].is_active)
				continue;
			if(var >= variable_names.size())
				throw std::runtime_error{"open_csv() Too much values for data row"};
			
			T val{};
			if(element.size()){
				float v;
				auto parse_res = std::from_chars(element.begin(), element.end(), v);
				val = v;
				if(parse_res.ec != std::errc{}){	// parsing error -> exchnage for category
					std::string el(element);
					if(ret.attributes[var].categories.count(el) > 0)
						val = ret.attributes[var].categories[el];
					else{
						val = category_values[var];
						category_values[var] += 1;
						ret.attributes[var].categories[el] = val;
					}
				}
			}

			if(val > ret.attributes[var].bounds.read().max)
				ret.attributes[var].bounds().max = val;
			if(val < ret.attributes[var].bounds.read().min)
				ret.attributes[var].bounds().min = val;
			ret.data.columns[var].push_back(val);
		}
	}
	// lexicographically ordering categorical data (using the automatical sorting provided by map)
	for(int var: util::size_range(ret.attributes)){
		auto& categories = ret.attributes[var].categories;
		if(categories.empty())
			continue;
		std::map<T, T> category_conversion;
		uint32_t counter{};
		for(auto& [category, value]: categories){
			T val = counter++;
			category_conversion[value] = val;
			value = val;
		}
		for(T& f: ret.data.columns[var])
			f = category_conversion[f];
	}

	for(int var: util::size_range(ret.attributes)){
		ret.attributes[var].id = variable_names[var];
		ret.attributes[var].display_name = variable_names[var];
		ret.attributes[var].data_bounds = ret.attributes[var].bounds.read();
	}
	if(query_attributes.size()){
		std::set<std::string_view> active_attributes;
		for(const auto& a: query_attributes)
			if(a.is_active)
				active_attributes.insert(a.id);
		erase_if(ret.attributes, [&](const structures::attribute& att){return active_attributes.count(att.id) == 0;});
	}
	erase_if(ret.data.columns, [](const std::vector<T>& v){return v.empty();});
	ret.data.dimension_sizes = {static_cast<uint32_t>(ret.data.columns[0].size())};
	if(query_attributes.size())
		ret.data.subsampleTrim({static_cast<uint32_t>(query_attributes.back().dimension_subsample)}, {{0, ret.data.dimension_sizes[0]}});
	ret.data.compress();
	
	return std::move(ret);
}
load_result<half> open_combined(std::string_view folder, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
	return {};
}
load_result<uint32_t> open_combined_compressed(std::string_view folder, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
	return {};
}
}

globals::dataset_t open_dataset(std::string_view filename, memory_view<structures::query_attribute> query_attributes, data_type_preference data_type_pref){
	// this method will open only a single dataset. If all datasets from a folder should be opened, handle the readout of all datasets outside this function
	// compressed data will be read nevertheless

	globals::dataset_t dataset;
	std::string_view file = filename.substr(filename.find_last_of("/\\") + 1);
	std::string_view file_extension = file.substr(file.find_last_of("."));
	dataset().id = file;
	switch(data_type_pref){
	case data_type_preference::float_precision:
		if(file_extension.empty()){
			throw std::runtime_error{"open_dataset() opening folder data not yet supported"};
		}
		else if(file_extension == ".nc"){
			auto data = open_internals::open_netcdf_float(filename, query_attributes);
			dataset().attributes = data.attributes;
			dataset().float_data = std::move(data.data);
		}
		else if(file_extension == ".csv"){
			auto data = open_internals::open_csv_float(filename, query_attributes);
			dataset().attributes = data.attributes;
			dataset().float_data = std::move(data.data);
		}
		break;
	case data_type_preference::half_precision:
		if(file_extension.empty()){
			throw std::runtime_error{"open_dataset() opening folder data not yet supported"};
		}
		else if(file_extension == ".nc"){
			auto data = open_internals::open_netcdf_half(filename, query_attributes);
			dataset().attributes = data.attributes;
			dataset().half_data = std::move(data.data);
		}
		else if(file_extension == ".csv"){
			auto data = open_internals::open_csv_half(filename, query_attributes);
			dataset().attributes = data.attributes;
			dataset().half_data = std::move(data.data);
		}
		break;
	default:
		throw std::runtime_error{"[warning] open_dataset() unrecognized data_type_preference"};
	}
	return std::move(dataset);
}

void convert_dataset(const structures::dataset_convert_data& convert_data){
    // TODO implement
}
}
}