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
        std::string         name;
        std::vector<int>    dependant_dimensions;
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
            //categorical data
            {
                throw std::runtime_error("netcdf_file::read_variable() categorical data currently can not be read.");
            }
            break;
            default:
                throw std::runtime_error("netcdf_file::read_variable() Unknown variable type");
        }

        return {ret, fill};
    }
};
}