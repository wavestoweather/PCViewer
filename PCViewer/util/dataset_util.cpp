#include "dataset_util.hpp"
#include <netcdf_file.hpp>
#include <c_file.hpp>
#include <sstream>
#include <filesystem>
#include <charconv>
#include <robin_hood.h>
#include <functional>

namespace util{
namespace dataset{
namespace open_internals{
template<>
load_result<float> open_netcdf(std::string_view filename, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
    structures::netcdf_file netcdf(filename);
	auto variables = netcdf.get_variable_infos();
	for(int var: util::size_range(variables)){
		if(query_attributes[var].id != variables[var].name)
			throw std::runtime_error{"open_netcdf() Attributes of the attribute query are not consistent with the netcdf file"};
	}
	load_result<float> ret{};
	for(const auto& d: netcdf.get_dimension_infos())
		ret.data.dimension_sizes.push_back(d.size);
	
	for(int var: util::size_range(query_attributes)){
		if(!query_attributes[var].is_active)
			continue;
		ret.data.column_dimensions.push_back(std::vector<uint32_t>(netcdf.get_variable_infos()[var].dependant_dimensions.begin(), netcdf.get_variable_infos()[var].dependant_dimensions.end()));
		auto [data, fill_value] = netcdf.read_variable<float>(var);
		ret.data.columns.push_back(std::move(data));
		ret.fill_values.push_back(fill_value);
	}

	return std::move(ret);
}
template<>
load_result<half> open_netcdf(std::string_view filename, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
    structures::netcdf_file netcdf(filename);
	auto variables = netcdf.get_variable_infos();
	for(int var: util::size_range(variables)){
		if(query_attributes[var].id != variables[var].name)
			throw std::runtime_error{"open_netcdf() Attributes of the attribute query are not consistent with the netcdf file"};
	}
	load_result<half> ret{};
	for(const auto& d: netcdf.get_dimension_infos())
		ret.data.dimension_sizes.push_back(d.size);
	
	for(int var: util::size_range(query_attributes)){
		if(!query_attributes[var].is_active)
			continue;
		ret.data.column_dimensions.push_back(std::vector<uint32_t>(netcdf.get_variable_infos()[var].dependant_dimensions.begin(), netcdf.get_variable_infos()[var].dependant_dimensions.end()));
		auto [data, fill_value] = netcdf.read_variable<half>(var);
		ret.data.columns.push_back(std::move(data));
		ret.fill_values.push_back(fill_value);
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
	ret.min_max_values.resize(variable_names.size());
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
					if(ret.categories[var].count(el) > 0)
						val = ret.categories[var][el];
					else{
						val = category_values[var];
						category_values[var] += 1;
						ret.categories[var][el] = val;
					}
				}
			}

			if(val > ret.min_max_values[var].max)
				ret.min_max_values[var].max = val;
			if(val < ret.min_max_values[var].min)
				ret.min_max_values[var].min = val;
			ret.data.columns[var].push_back(val);
		}
	}
	// lexicographically ordering categorical data (using the automatical sorting provided by map)
	for(auto& [var, categories]: ret.categories){
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
	ret.min_max_values.resize(variable_names.size());
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
					if(ret.categories[var].count(el) > 0)
						val = ret.categories[var][el];
					else{
						val = category_values[var];
						category_values[var] += 1;
						ret.categories[var][el] = val;
					}
				}
			}

			if(val > ret.min_max_values[var].max)
				ret.min_max_values[var].max = val;
			if(val < ret.min_max_values[var].min)
				ret.min_max_values[var].min = val;
			ret.data.columns[var].push_back(val);
		}
	}
	// lexicographically ordering categorical data (using the automatical sorting provided by map)
	for(auto& [var, categories]: ret.categories){
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
	return {};
}

void convert_dataset(const structures::dataset_convert_data& convert_data){
    // TODO implement
}
}
}