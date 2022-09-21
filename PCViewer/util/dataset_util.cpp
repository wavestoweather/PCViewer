#include "dataset_util.hpp"
#include <netcdf_file.hpp>

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
		ret.data.dimensionSizes.push_back(d.size);
	
	for(int var: util::size_range(query_attributes)){
		if(!query_attributes[var].is_active)
			continue;
		ret.data.columnDimensions.push_back(std::vector<uint32_t>(netcdf.get_variable_infos()[var].dependant_dimensions.begin(), netcdf.get_variable_infos()[var].dependant_dimensions.end()));
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
		ret.data.dimensionSizes.push_back(d.size);
	
	for(int var: util::size_range(query_attributes)){
		if(!query_attributes[var].is_active)
			continue;
		ret.data.columnDimensions.push_back(std::vector<uint32_t>(netcdf.get_variable_infos()[var].dependant_dimensions.begin(), netcdf.get_variable_infos()[var].dependant_dimensions.end()));
		auto [data, fill_value] = netcdf.read_variable<half>(var);
		ret.data.columns.push_back(std::move(data));
		ret.fill_values.push_back(fill_value);
	}

	return std::move(ret);
}
template<>
load_result<float> open_csv(std::string_view filenmae, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
	return {};
}
template<>
load_result<half> open_csv(std::string_view filenmae, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
    return {};
}
load_result<half> open_combined(std::string_view folder, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
	return {};
}
load_result<uint32_t> open_combined_compressed(std::string_view folder, memory_view<structures::query_attribute> query_attributes, const load_information* partial_info){
	return {};
}
}

globals::dataset_t open_dataset(std::string_view filename, memory_view<structures::query_attribute> query_attributes){
	return {};
}

void convert_dataset(const structures::dataset_convert_data& convert_data){
    // TODO implement
}
}
}