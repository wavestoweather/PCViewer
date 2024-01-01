#pragma once

#include <netcdf.h>
#include <string_view>
#include <stdexcept>
#include <vector>
#include <ranges.hpp>
#include <optional>

namespace structures{
class netcdf_file{
    int _file_handle{};
public:
    int dimension_count{};
    int variable_count{};
    int ngatts{};
    int unlimited_dimension{};
    struct var_info{
        std::string         name{};
        std::vector<int>    dependant_dimensions{};
        nc_type             variable_type{};
        float               offset{};
        float               scale{1.f};
    };
    struct dim_info{
        std::string name;
        size_t      size;
    };
private:
    std::vector<var_info> _variable_infos;
    std::vector<dim_info> _dimension_infos;
public:
    netcdf_file() = delete;
    netcdf_file(std::string_view path, int mode = NC_NOWRITE) {
        int res = nc_open(path.data(), mode, &_file_handle);
        if(res)
            throw std::runtime_error{"netcdf_file(...) Failed to open file " + std::string(path)};
        res = nc_inq(_file_handle, &dimension_count, &variable_count, &ngatts, &unlimited_dimension);
        if(res)
            throw std::runtime_error("netcdf_file(..) Failed to read variables and dimension information");
    }
    ~netcdf_file() {nc_close(_file_handle);}

    const std::vector<var_info>& get_variable_infos(){
        if(_variable_infos.size())
            return _variable_infos;
        _variable_infos.resize(variable_count);
        for(int i: util::i_range(variable_count)){
            _variable_infos[i].name.resize(NC_MAX_NAME, 0);
            int res = nc_inq_varname(_file_handle, i, _variable_infos[i].name.data());
            _variable_infos[i].name = _variable_infos[i].name.substr(0, _variable_infos[i].name.find('\0'));
            if(res)
                throw std::runtime_error("netcdf_file::get_variable_infos() Failed to get variable name");
            int dims;
            res = nc_inq_varndims(_file_handle, i, &dims);
            if(res)
                throw std::runtime_error("netcdf_file::get_variable_infos() Failed to get variable dimension size");
            _variable_infos[i].dependant_dimensions.resize(dims);
            res = nc_inq_vardimid(_file_handle, i, _variable_infos[i].dependant_dimensions.data());
            if(res)
                throw std::runtime_error("netcdf_file::get_variable_infos() Failed to get variable dimensions");
            _variable_infos[i].variable_type = var_type(i);
            nc_type att_type;
            res = nc_inq_att(_file_handle, i, "scale_factor", &att_type, {});
            if(res == NC_NOERR){
                switch(att_type){
                    case NC_FLOAT: nc_get_att_float(_file_handle, i, "scale_factor", &_variable_infos[i].scale); break;
                    case NC_DOUBLE: {double v; if(nc_get_att_double(_file_handle, i, "scale_factor", &v)) break; _variable_infos[i].scale = static_cast<float>(v); break;}
                    case NC_SHORT: {int16_t v; if(nc_get_att_short(_file_handle, i, "scale_factor", &v)) break; _variable_infos[i].scale = static_cast<float>(v); break;}
                    default:
                        ::logger << logging::warning_prefix << " Attribute type to parse scale_factor not implemented" << logging::endl;
                }
            }
            res = nc_inq_att(_file_handle, i, "add_offset", &att_type, {});
            if(res == NC_NOERR){
                switch(att_type){
                    case NC_FLOAT: nc_get_att_float(_file_handle, i, "add_offset", &_variable_infos[i].offset); break;
                    case NC_DOUBLE: {double v; if(nc_get_att_double(_file_handle, i, "add_offset", &v)) break; _variable_infos[i].offset = static_cast<float>(v); break;}
                    case NC_SHORT: {int16_t v; if(nc_get_att_short(_file_handle, i, "add_offset", &v)) break; _variable_infos[i].offset = static_cast<float>(v); break;}
                    default:
                        ::logger << logging::warning_prefix << " Attribute type to parse add_offset not implemented" << logging::endl;
                }
            }
        }
        return _variable_infos;
    }
    
    const std::vector<dim_info>& get_dimension_infos(){
        if(_dimension_infos.size())
            return _dimension_infos;
        _dimension_infos.resize(dimension_count);
        for(int i: util::i_range(dimension_count)){
            _dimension_infos[i].name.resize(NC_MAX_NAME);
            int res = nc_inq_dim(_file_handle, i, _dimension_infos[i].name.data(), &_dimension_infos[i].size);
            if(res)
                throw std::runtime_error("netcdf_file::get_dimension_infos() Failed to get dimensions");
            _dimension_infos[i].name.shrink_to_fit();
        }
        return _dimension_infos;
    }

    // care the size here is calculated including stringlength dimensions
    size_t data_size() {
        if(_dimension_infos.empty())
            get_dimension_infos();
        
        size_t size = 1;
        for(const auto& dim_info: _dimension_infos)
            size *= dim_info.size;
        return size;
    }

    nc_type var_type(int variable) const {
        nc_type type;
        int res = nc_inq_vartype(_file_handle, variable, &type);
        if(res)
            throw std::runtime_error("netcdf_file::var_type() Failed to get variable type");
        return type;
    }

    template<class T>
    std::tuple<std::vector<T>, std::optional<T>> read_variable(int variable){
        if(_variable_infos.empty())
            get_variable_infos();
        if(_dimension_infos.empty())
            get_dimension_infos();
        size_t data_size{1};
        for(int dim: _variable_infos[variable].dependant_dimensions)
            data_size *= _dimension_infos[dim].size;
        std::vector<T> ret;
        std::optional<T> fill;
        int has_fill{};
        nc_type type;
        int res = nc_inq_vartype(_file_handle, variable, &type);
        if(res)
            throw std::runtime_error("netcdf_file::read_variable() Failed to get variable type");
        switch(type){
            case NC_FLOAT:{
                std::vector<float> vals(data_size);
                res = nc_get_var_float(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                float f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, float>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_DOUBLE:
            {
                std::vector<double> vals(data_size);
                res = nc_get_var_double(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                double f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, double>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_INT:
            {
                std::vector<int> vals(data_size);
                res = nc_get_var_int(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                int f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, int>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_UINT:
            {
                std::vector<uint32_t> vals(data_size);
                res = nc_get_var_uint(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                uint32_t f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, uint32_t>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_UINT64:{
                std::vector<unsigned long long> vals(data_size);
                res = nc_get_var_ulonglong(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                unsigned long long f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, unsigned long long>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_INT64:{
                std::vector<long long> vals(data_size);
                res = nc_get_var_longlong(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                long long f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, long long>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_UBYTE:
            case NC_BYTE:{
                std::vector<uint8_t> vals(data_size);
                res = nc_get_var_ubyte(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable values");
                uint8_t f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, uint8_t>)
                    ret = std::move(vals);
                else
                    ret = std::vector<T>(vals.begin(), vals.end());
                if(has_fill != 1)
                    fill = {static_cast<T>(f)};
                break;
            }
            case NC_CHAR:
            {
                // strings can be stored in this format, they then require 2 dims: one containing the amount of string, and one depicting the max string length
                if constexpr (!std::is_same_v<T, char>)
                    throw std::runtime_error{"netcdf_file::read_variable() Char variable has to be read out with read_variable<char>()"};
                std::vector<char> vals(data_size);
                res = nc_get_var_text(_file_handle, variable, vals.data());
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to variable values");
                
                char f;
                res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
                if(res)
                    throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");
                
                if constexpr (std::is_same_v<T, char>){
                    ret = std::move(vals);
                    if(has_fill)
                        fill = f;
                }
                break;
            }
            case NC_STRING:
            {
                // string in netcdf is a vector of char*, thus real c-strings
                if constexpr (!std::is_same_v<T, std::string_view>)
                    throw std::runtime_error{"netcdf_file::read_variable() String variable has to be read out with read_variable<std::string_view>()"};
            }
            break;
            default:
                throw std::runtime_error("netcdf_file::read_variable() Unknown variable type");
        }

        return {ret, fill};
    }
    // string_view overload to avoid problems with conversion
    std::tuple<std::vector<std::string_view>, std::optional<std::string_view>> read_variable(int variable){
        if(_variable_infos.empty())
            get_variable_infos();
        if(_dimension_infos.empty())
            get_dimension_infos();
        size_t data_size{1};
        for(int dim: _variable_infos[variable].dependant_dimensions)
            data_size *= _dimension_infos[dim].size;
        std::vector<std::string_view> ret;
        std::optional<std::string_view> fill;
        int has_fill{};
        nc_type type;
        int res = nc_inq_vartype(_file_handle, variable, &type);
        if(res)
            throw std::runtime_error("netcdf_file::read_variable() Failed to get variable type");

        if(type != NC_STRING)
             throw std::runtime_error("netcdf_file::read_variable() Tried to read non string variable with read_variable<std::string_view>()");

        std::vector<char*> vals(data_size);
        res = nc_get_var_string(_file_handle, variable, vals.data());
        if(res)
            throw std::runtime_error("netcdf_file::read_variable() Failed to variable values");
        
        char* f;
        res = nc_inq_var_fill(_file_handle, variable, &has_fill, &f);
        if(res)
            throw std::runtime_error("netcdf_file::read_variable() Failed to get variable fill value");

        ret.resize(data_size);
        for(auto&& [e, i]: util::enumerate(ret))
            e = std::string_view(vals[i]);
        if(has_fill)
            fill = std::string_view(f);
        return {ret, fill};
    }
};
}